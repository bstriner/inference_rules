import csv
import os

import keras.backend as K
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from .adjacency import calc_adjacency
from .gumbel import gumbel_sigmoid, gumbel_softmax
from .gumbel_np import sample_argmax_np
from .parser import entity_types
from .tensor_util import tensor_one_hot, load_latest_weights
from .util import split_data, make_dir


class InferenceModel(object):
    def __init__(self,
                 train,
                 valid,
                 rule_counts,
                 entities,
                 relations,
                 opt,
                 initializer,
                 tau0=5.,
                 tau_min=0.25,
                 tau_decay=1e-6,
                 srng=RandomStreams(123),
                 splits=10,
                 eps=1e-9):
        rule_depth = len(rule_counts)
        self.rule_counts = rule_counts
        self.rule_depth = rule_depth
        r_k = len(relations)
        self.r_k = r_k

        # Inputs
        walks = [T.imatrix(name='walks{}'.format(i)) for i in range(rule_depth)]
        walk_assignments = [T.ivector(name='assignment{}'.format(i)) for i in range(rule_depth)]
        # walk_assignments: (wn,) int [0-sample_count]
        sample_queries = T.ivector(name='queries') # (sample_count,)
        targets = T.ivector(name='targets') # (sample_count,)
        target_weights = T.fvector(name='weights') # (sample_count,)
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
        param_alphas = [K.variable(initializer((depth+1, rule_n, r_k*2)), name='alphas{}'.format(depth))
                       for depth,rule_n in enumerate(rule_counts)]
        param_dists = [K.variable(initializer((rule_n, r_k, 2)), name='param_dist')
                       for depth,rule_n in enumerate(rule_counts)]
        param_bias = K.variable(initializer((r_k, 2)), name='bias')
        params = param_alphas + param_dists + [param_bias]

        # Rules
        alphas = [gumbel_sigmoid(param_alphas[i], srng=srng, temperature=tau) # (rd, rn, rk*2)
                  for i in range(rule_depth)]
        dists = [T.exp(param_dists[i]) for i in range(rule_depth)]  # (rule_n, r_k, 2)
        bias = T.exp(param_bias) # (r_k, 2)
        self.alphas = alphas
        self.dists = dists
        self.bias = bias

        # Features
        feats = self.calc_features(walks=walks,
                                   assignments=walk_assignments,
                                   alphas=alphas,
                                   dists=dists,
                                   queries=walk_queries,
                                   sample_count=sample_count) # (sample_count, 2)
        bias_vals = bias

        # Loss
        expected_p = feats[:,0] / (eps+T.sum(feats, axis=1))

        psel = T.switch(targets, expected_p, 1.-expected_p) # (sample_count,)
        ploss = -T.log(eps+psel) # (sample_count,)
        weighted_loss = T.sum(ploss * target_weights) # scalar
        total_loss = weighted_loss

        # accuracy
        phard = T.gt(expected_p, 0.5)  # (n,)
        acc = T.mean(T.eq(phard, targets))  # scalar

        updates = opt.get_updates(total_loss, params)
        self.function_train = theano.function([walks, sample_queries, walk_assignments, targets, target_weights],
                                              [total_loss, acc],
                                              updates=updates + iter_updates)

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
                w = walks[depth] # (wn, rd) int 0-rk*2
                assignment = assignments[depth] # (wn,) int 0-wn
                alpha = alphas[depth] # (rn, rd, rk*2)
                dist = dists[depth] # (rn, rk, 2)
                query = queries[depth] # (wn,) int 0-rk
                wn = w.shape[0]
                # get alphas at those relationships
                idx_rd = T.mgrid[:wn, :depth+1][1]
                idx_rk = w
                alpha_sel = alpha[:,idx_rd,idx_rk] # (rn, wn, rd)
                alpha_prod = T.prod(alpha_sel, axis=2) # (rn, wn)
                # get distributions by query
                dist_sel = dist[:,query,:] # (rn, wn, 2)
                # calculate features
                rule_scores = T.sum(dist_sel * (alpha_prod.dimshuffle((0,1,'x'))), axis=0) # (wn, 2)
                assignment_one_hot = tensor_one_hot(assignment, k=sample_count) # (wn, sample_count)
                sample_features = T.dot(T.transpose(assignment_one_hot, (1,0)), rule_scores) # (sample_count, 2)
                features += sample_features
        return features

    def calc_p(self, s, r, t, n, adj):

        m0 = tensor_one_hot(s, k=self.e_k, dtype='float32')  # (n, e_k)
        memory = T.repeat(T.reshape(m0, (1, n, 1, self.e_k)), axis=0, repeats=self.rule_n)  # (rule_n, n,  1, e_k)
        for i in range(self.rule_depth):
            alpha = self.alphas[i, :, :]  # (rule_n, r_k)
            if i == 0:
                msel = memory[:, :, 0, :]  # (n, rule_n, e_k)
            else:
                beta = self.betas[i - 1]  # (rule_n, memories)
                msel = T.sum(memory * (beta.dimshuffle((0, 'x', 1, 'x'))), axis=2)  # (rule_n, n, e_k)
            mat = T.sum((adj.dimshuffle(('x', 0, 1, 2))) * (alpha.dimshuffle((0, 1, 'x', 'x'))),
                        axis=1)  # (rule_n, e_k, e_k)
            y = T.batched_dot(msel, mat)  # (rule_n, n, e_k)
            y = T.reshape(y, (self.rule_n, n, 1, self.e_k))
            memory = T.concatenate((memory, y), axis=2)  # (rule_n, n, memories, e_k)
        beta = self.betas[-1]  # (rule_n, memories)

        # combine prediction
        pred = T.sum(memory * (beta.dimshuffle((0, 'x', 1, 'x'))), axis=2)  # (rule_n, n, e_k)
        p = T.dot(pred, t)  # (rule_n, n)
        dp = self.param_dist[:, r, :]  # (rule_n, n, 2)
        h = T.sum((p.dimshuffle((0, 1, 'x'))) * dp, 0)  # (n, 2)
        expected_p = h[:, 0] / T.sum(h, axis=1)  # (n,)
        return expected_p

    def sample_query(self,
                     holdout,
                     target_logits,
                     source_logits,
                     pos_samples,
                     neg_target_samples,
                     neg_source_samples):
        # sample positive triples from held-out set
        pos_idx = np.random.random_integers(low=0, high=holdout.shape[0] - 1, size=(pos_samples,))
        pos_triples = holdout[pos_idx]

        neg_samples = []
        for _ in range(neg_target_samples):
            # corrupt target
            target_p = target_logits[pos_triples[:, 1], :]
            neg_targets = sample_argmax_np(target_p)
            neg_target_triples = np.copy(pos_triples)
            neg_target_triples[:, 2] = neg_targets

        for _ in range(neg_source_samples):
            # corrupt source
            source_p = source_logits[pos_triples[:, 1], :]
            neg_sources = sample_argmax_np(source_p)
            neg_source_triples = np.copy(pos_triples)
            neg_source_triples[:, 2] = neg_sources

        neg_samples = np.concatenate(neg_samples, axis=0)

        return pos_triples, neg_samples

    def sample_query_data(self,
                          holdout,
                          target_logits,
                          source_logits,
                          pos_samples,
                          neg_target_samples,
                          neg_source_samples):
        pos, neg = self.sample_query(holdout=holdout,
                                     target_logits=target_logits,
                                     source_logits=source_logits,
                                     pos_samples=pos_samples,
                                     neg_target_samples=neg_target_samples,
                                     neg_source_samples=neg_source_samples)
        t0 = np.zeros((pos.shape[0],), dtype=np.int32)
        t1 = np.ones((neg.shape[0],), dtype=np.int32)

        d0 = np.concatenate((pos, neg), axis=0)
        d1 = np.concatenate((t0, t1), axis=0)
        return d0, d1

    def batch_data(self, pos_samples, neg_target_samples, neg_source_samples):
        split = np.random.random_integers(low=0, high=self.splits - 1)
        d0, d1 = self.sample_query_data(holdout=self.train_holdout[split],
                                        target_logits=self.train_candidates_target,
                                        source_logits=self.train_candidates_source,
                                        pos_samples=pos_samples,
                                        neg_target_samples=neg_target_samples,
                                        neg_source_samples=neg_source_samples)
        return split, d0, d1

    def batch_data_val(self, pos_samples, neg_target_samples, neg_source_samples):
        d0, d1 = self.sample_query_data(holdout=self.valid,
                                        target_logits=self.val_candidates_target,
                                        source_logits=self.val_candidates_source,
                                        pos_samples=pos_samples,
                                        neg_target_samples=neg_target_samples,
                                        neg_source_samples=neg_source_samples)
        return d0, d1

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

    def train(self,
              output_path,
              epochs,
              batches,
              val_batches,
              pos_samples,
              neg_target_samples,
              neg_source_samples):
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
                        x = self.batch_data(pos_samples=pos_samples,
                                            neg_target_samples=neg_target_samples,
                                            neg_source_samples=neg_source_samples)
                        rets = self.function_train(*x)
                        for i in range(2):
                            data[i].append(rets[i])
                        it.desc = 'Epoch {} NLL {:.04f} ACC {:.04f}'.format(
                            epoch,
                            np.asscalar(np.mean(data[0])),
                            np.asscalar(np.mean(data[1]))
                        )
                    nll = np.asscalar(np.mean(data[0]))
                    acc = np.asscalar(np.mean(data[1]))
                    val_nll, val_acc = self.validate(batches=val_batches,
                                                     pos_samples=pos_samples,
                                                     neg_target_samples=neg_target_samples,
                                                     neg_source_samples=neg_source_samples)
                    w.writerow([epoch, nll, acc, np.asscalar(val_nll), np.asscalar(val_acc)])
                    f.flush()
