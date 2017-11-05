import csv
import itertools
import os

import keras.backend as K
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from .gumbel import gumbel_sigmoid
from .tensor_util import tensor_one_hot, load_latest_weights
from .util import make_dir


class InferenceWalkerModel(object):
    def __init__(self,
                 rule_counts,
                 relations,
                 opt,
                 initializer,
                 regularizer=None,
                 tau0=5.,
                 tau_min=0.25,
                 tau_decay=1e-6,
                 srng=RandomStreams(123),
                 eps=1e-9):
        rule_depth = len(rule_counts)
        self.rule_counts = rule_counts
        self.rule_depth = rule_depth
        self.relations = relations
        self.r_k = len(relations)

        # Inputs
        walks = [T.imatrix(name='walks{}'.format(i)) for i in range(rule_depth)]
        walk_assignments = [T.ivector(name='assignment{}'.format(i)) for i in range(rule_depth)]
        # walk_assignments: (wn,) int [0-sample_count]
        sample_queries = T.ivector(name='queries')  # (sample_count,)
        targets = T.ivector(name='targets')  # (sample_count,)
        target_weights = T.fvector(name='weights')  # (sample_count,)
        sample_count = targets.shape[0]
        walk_queries = [sample_queries[walk_assignments[i]] for i in range(rule_depth)]

        # Temperature
        self.iteration = K.variable(0, dtype='int32', name='iteration')
        iter_updates = [(self.iteration, self.iteration + 1)]
        tau = T.constant(tau0, dtype='float32', name='tau0')
        if tau_decay > 0:
            tau_decay = T.constant(tau_decay, name='tau_decay', dtype='float32')
            tau_min = T.constant(tau_min, name='tau_min', dtype='float32')
            tau = tau / (1. + (tau_decay * self.iteration))
            tau = T.nnet.relu(tau - tau_min) + tau_min
        self.tau = tau

        # Parameters
        param_alphas = [K.variable(initializer((depth + 1, rule_n, self.r_k * 2)), name='alphas{}'.format(depth))
                        for depth, rule_n in enumerate(rule_counts)]
        param_dists = [K.variable(initializer((rule_n, self.r_k, 2)), name='param_dist')
                       for depth, rule_n in enumerate(rule_counts)]
        param_bias = K.variable(initializer((self.r_k, 2)), name='bias')
        params = param_alphas + param_dists + [param_bias]

        # Rules
        alphas = [gumbel_sigmoid(param_alphas[i], srng=srng, temperature=tau)  # (rd, rn, rk*2)
                  for i in range(rule_depth)]
        dists = [T.exp(param_dists[i]) for i in range(rule_depth)]  # (rule_n, r_k, 2)
        bias = T.exp(param_bias)  # (r_k, 2)
        self.alphas = alphas
        self.dists = dists
        self.bias = bias
        reg_loss = T.constant(0.)
        if regularizer:
            for a in alphas:
                reg_loss += regularizer(a)
            for d in dists:
                reg_loss += regularizer(d)

        # Features
        feats = self.calc_features(walks=walks,
                                   assignments=walk_assignments,
                                   alphas=alphas,
                                   dists=dists,
                                   queries=walk_queries,
                                   sample_count=sample_count)  # (sample_count, 2)
        ybias = bias[sample_queries, :]  # (sample_count, 2)
        ydist = feats + ybias

        # Loss
        expected_p = ydist[:, 0] / (eps + T.sum(ydist, axis=1))

        psel = T.switch(targets, expected_p, 1. - expected_p)  # (sample_count,)
        ploss = -T.log(eps + psel)  # (sample_count,)
        weighted_loss = T.sum(ploss * target_weights)  # scalar
        total_loss = weighted_loss + reg_loss

        # accuracy
        phard = T.gt(expected_p, 0.5)  # (n,)
        acc = T.mean(T.eq(phard, targets))  # scalar

        updates = opt.get_updates(total_loss, params)
        self.function_train = theano.function([walks, sample_queries, walk_assignments, targets, target_weights],
                                              [total_loss, acc],
                                              updates=updates + iter_updates)

        val_alphas = [T.gt(param_alphas[i], 0.5)  # (rd, rn, rk*2)
                      for i in range(rule_depth)]
        self.function_get_rules = theano.function([], val_alphas + dists + [bias])

        # validation
        """
        val_expected_p = self.calc_p(s=s, r=r, t=t, n=n, adj=train_adj_all)
        val_psel = T.switch(input_targets, val_expected_p, 1. - val_expected_p)
        val_nll = -T.log(eps + val_psel)
        val_phard = T.gt(val_expected_p, 0.5)  # (n,)
        val_acc = T.mean(T.eq(val_phard, input_targets))  # scalar
        self.function_val = theano.function([input_triples, input_targets], [val_nll, val_acc])
        """
        self.weights = params + opt.weights

    def calc_features(self, walks, assignments, alphas, dists, queries, sample_count):
        features = T.zeros((sample_count, 2))
        for depth in range(self.rule_depth):
            rn = self.rule_counts[depth]
            if rn > 0:
                w = walks[depth]  # (wn, rd) int 0-rk*2
                assignment = assignments[depth]  # (wn,) int 0-wn
                alpha = alphas[depth]  # (rn, rd, rk*2)
                dist = dists[depth]  # (rn, rk, 2)
                query = queries[depth]  # (wn,) int 0-rk
                wn = w.shape[0]
                # get alphas at those relationships
                idx_rd = T.mgrid[:wn, :depth + 1][1]
                idx_rk = w
                alpha_sel = alpha[:, idx_rd, idx_rk]  # (rn, wn, rd)
                alpha_prod = T.prod(alpha_sel, axis=2)  # (rn, wn)
                # get distributions by query
                dist_sel = dist[:, query, :]  # (rn, wn, 2)
                # calculate features
                rule_scores = T.sum(dist_sel * (alpha_prod.dimshuffle((0, 1, 'x'))), axis=0)  # (wn, 2)
                assignment_one_hot = tensor_one_hot(assignment, k=sample_count)  # (wn, sample_count)
                sample_features = T.dot(T.transpose(assignment_one_hot, (1, 0)), rule_scores)  # (sample_count, 2)
                features += sample_features
        return features

    """
    def validate(self, batches, pos_samples, neg_target_samples, neg_source_samples):
        data = [[] for _ in range(2)]
        for _ in range(batches):
            x = self.batch_data_val(pos_samples=pos_samples,
                                    neg_target_samples=neg_target_samples,
                                    neg_source_samples=neg_source_samples)
            rets = self.function_val(*x)
            for i in range(2):
                data[i].append(rets[i])
        return np.mean(data[0]), np.mean(data[1])
    """

    def train(self,
              output_path,
              train_walks,
              epochs,
              batches,
              val_batches,
              batch_size=128):
        make_dir(output_path)
        initial_epoch = load_latest_weights(output_path, r'model-(\d+).h5', self.weights)
        if initial_epoch < epochs:
            with open(os.path.join(output_path, 'history.csv'), 'ab') as f:
                w = csv.writer(f)
                w.writerow(['Epoch', 'NLL', 'Acc', 'Val NLL', 'Val ACC'])
                f.flush()
                for epoch in tqdm(range(initial_epoch, epochs), desc='Training'):
                    it = tqdm(range(batches), desc='Epoch {}'.format(epoch))
                    data = [[] for _ in range(2)]
                    for _ in it:
                        rets = self.train_batch(train_walks, batch_size)
                        for i in range(2):
                            data[i].append(rets[i])
                        it.desc = 'Epoch {} NLL {:.04f} ACC {:.04f}'.format(
                            epoch,
                            np.asscalar(np.mean(data[0])),
                            np.asscalar(np.mean(data[1]))
                        )
                    nll = np.asscalar(np.mean(data[0]))
                    acc = np.asscalar(np.mean(data[1]))
                    # val_nll, val_acc = self.validate(batches=val_batches,
                    #                                 pos_samples=pos_samples,
                    #                                 neg_target_samples=neg_target_samples,
                    #                                 neg_source_samples=neg_source_samples)
                    w.writerow([epoch, nll, acc])  # , np.asscalar(val_nll), np.asscalar(val_acc)])
                    f.flush()
                    self.write_rules('{}/rules-{:08d}.csv'.format(output_path, epoch))

    def train_batch(self, walk_data, batch_size):
        data = self.sample_batch(walk_data, batch_size)
        return self.function_train(*data)

    def sample_batch(self, walk_data, batch_size):
        # [walks, sample_queries, walk_assignments, targets, target_weights]
        walks = []
        sample_queries = []
        walk_assignments = []
        targets = []
        target_weights = []

        n = len(walk_data)
        idx = np.random.choice(np.arange(n), size=(batch_size,), replace=False)
        for i in idx:
            w = walk_data[i]
            # positive sample
            walks.append(w['rels'])
            sample_queries.append(w['r'])
            walk_assignments.append((2 * i) * np.ones((w['rels'].shape[0],)))
            targets.append(1)
            target_weights.append(1)
            # negative sample
            nsidx = np.random.random_integers(low=0, high=len(w['neg_walks']) - 1)
            ns = w['neg_walks'][nsidx]
            walks.append(ns['rels'])
            sample_queries.append(w['r'])
            walk_assignments.append(((2 * i) + 1) * np.ones((w['rels'].shape[0],)))
            targets.append(0)
            target_weights.append(1)

        walks = np.concatenate(walks, axis=0)
        sample_queries = np.array(sample_queries, dtype=np.int32)
        walk_assignments = np.concatenate(walk_assignments, axis=0)
        targets = np.array(targets, dtype=np.int32)
        target_weights = np.array(target_weights, dtype=np.float32)
        return walks, sample_queries, walk_assignments, targets, target_weights

    def enumerate_rules(self):
        rules = []
        rule_data = self.function_get_rules()
        counts = np.array(self.rule_counts)
        offsets = np.cumsum(counts) - counts
        cutoff = 1.
        for d in range(len(self.rule_counts)):
            offset = offsets[d]
            rule_count = self.rule_counts[d]
            alphas = rule_data[d]  # ( depth, rule_count, r_k*2)
            dists = rule_data[d + len(self.rule_counts)]  # (rule_count, r_k, 2)
            for i in range(rule_count):
                rule_id = i + offset
                rels = [list(np.nonzero(alphas[j, i, :])[0]) for j in range(d + 1)]
                combos = itertools.product(*rels)
                for c in combos:
                    for r in range(self.r_k):
                        if dists[i, r, 0] > cutoff or dists[i, r, 1] > cutoff:
                            rule = {'rule_id': rule_id,
                                    'walk': c,
                                    'dist': dists[i, r, :],
                                    'r': r
                                    }
                            rules.append(rule)
        return rules

    def relationship_name(self, r):
        if r < len(self.relations):
            return self.relations[r]
        else:
            return "{}^-1".format(self.relations[r - len(self.relations)])

    def formula(self, walk):
        strs = []
        for i in range(len(walk)):
            a = "z{}".format(i - 1)
            b = "z{}".format(i)
            if i == 0:
                a = "s"
            if i == len(walk) - 1:
                b = "t"
            if walk[i] < len(self.relations):
                r = self.relations[walk[i]]
            else:
                r = self.relations[walk[i] - len(self.relations)]
                a, b = b, a
            strs.append("{}({},{})".format(r, a, b))
        return " && ".join(strs)

    def write_rules(self, output_path):
        rules = self.enumerate_rules()
        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(['Rule ID', 'Query', 'Rule', 'Positive Votes', 'Negative Votes'])
            for rule in rules:
                query = '{}(s,t)'.format(self.relations[rule['r']])
                rulestr = self.formula(rule['walk'])
                pos = rule['dist'][0]
                neg = rule['dist'][0]
                w.writerow([rule['rule_id'], query, rulestr, pos, neg])
