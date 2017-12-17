import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import RunConfig
from tensorflow.contrib.training import HParams
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs

from .gumbel_tf import gumbel_sigmoid

EPSILON = 1e-7


def calc_features(walks, assignments, alphas, dists, queries, sample_count, rule_depth):
    # if tf.flags.FLAGS.sigmoid_dist:
    #    features = tf.zeros((sample_count,), dtype=tf.float32)
    # else:
    #    features = tf.zeros((sample_count, 2), dtype=tf.float32)
    features = 0.
    for depth in range(rule_depth):
        w = walks[depth]  # (wn, rd) int 0-rk*2
        # assert w.ndim == 2
        assignment = assignments[depth]  # (wn,) int 0-wn
        # assert assignment.ndim == 1
        alpha = alphas[depth]  # (rn, rd, rk*2)
        # assert alpha.ndim == 3
        dist = dists[depth]  # (rn, rk, 2)
        # assert dist.ndim == 3
        query = queries[depth]  # (wn,) int 0-rk
        # assert query.ndim == 1
        wn = tf.shape(w)[0]
        rn = tf.shape(alpha)[0]
        # get alphas at those relationships
        idx_rd = tf.meshgrid(tf.range(rn), tf.range(wn), tf.range(depth + 1), indexing='ij')
        idx_w = tf.tile(tf.expand_dims(w, axis=0), multiples=(rn, 1, 1))
        ids = [idx_rd[0], idx_rd[1], idx_w]
        ids = tf.stack(ids, axis=3)
        alpha_sel = tf.gather_nd(alpha, ids)  # alpha[ids[0], ids[1], ids[2]]  # (rn, wn, rd)
        if tf.flags.FLAGS.sigmoid_dist:
            alpha_prod = tf.reduce_prod(alpha_sel, axis=2)  # (rn, wn)
            dist_sel = tf.gather(dist, query, axis=1)  # (rn, wn)
            rule_scores = tf.reduce_sum(dist_sel * alpha_prod, axis=0)  # (wn,)
            assignment_one_hot = tf.one_hot(assignment, sample_count)  # (wn, sample_count)
            sample_features = tf.tensordot(rule_scores, assignment_one_hot, axes=(0, 0))  # (sample_count,)
        else:
            alpha_prod = tf.reduce_prod(alpha_sel, axis=2, keep_dims=True)  # (rn, wn, 1)
            dist_sel = tf.gather(dist, query, axis=1)  # (rn, wn, 2)
            rule_scores = tf.reduce_sum(dist_sel * alpha_prod, axis=0)  # (wn, 2)
            assignment_one_hot = tf.one_hot(assignment, sample_count)  # (wn, sample_count)
            sample_features = tf.matmul(tf.transpose(assignment_one_hot, (1, 0)), rule_scores)  # (sample_count, 2)

        features += sample_features
    return features


from tensorflow.python.ops.init_ops import RandomUniform, Zeros


def inference_fn(features, mode, params):
    rule_count = params.rule_count
    rule_depth = params.rule_depth
    r_k = params.r_k

    # Inputs
    walks = [features['walks_{}'.format(i)] for i in range(rule_depth)]
    walk_assignments = [features['assignments_{}'.format(i)] for i in range(rule_depth)]
    # walk_assignments: (wn,) int [0-sample_count]
    sample_queries = features['queries']
    # targets = features['targets']
    sample_count = tf.shape(sample_queries)[0]
    walk_queries = [tf.gather(sample_queries, walk_assignments[i]) for i in range(rule_depth)]

    # Temperature
    tau = tf.constant(params.tau0, dtype=tf.float32, name='tau0')
    if params.tau_decay > 0:
        tau_decay = tf.constant(params.tau_decay, name='tau_decay', dtype=tf.float32)
        tau_min = tf.constant(params.tau_min, name='tau_min', dtype=tf.float32)
        tau = tau / (1. + (tau_decay * tf.cast(tf.train.get_global_step(), tf.float32)))
        tau = tf.nn.relu(tau - tau_min) + tau_min
    tf.summary.scalar('tau', tau)

    # Parameters
    scale = 0.05
    initializer = RandomUniform(minval=-scale, maxval=scale)
    param_alphas = [tf.get_variable(name='alphas_{}'.format(depth), shape=[rule_count, depth + 1, r_k * 2],
                                    trainable=True, dtype=tf.float32, initializer=initializer)
                    for depth in range(rule_depth)]
    alphas = [gumbel_sigmoid(param_alphas[i], temperature=tau)  # (rd, rn, rk*2)
              for i in range(rule_depth)]

    if tf.flags.FLAGS.sigmoid_dist:
        param_dists = [tf.get_variable(name='param_dist_{}'.format(depth), shape=[rule_count, r_k],
                                       trainable=True, dtype=tf.float32, initializer=Zeros)
                       for depth in range(rule_depth)]
        param_bias = tf.get_variable(name='bias', shape=[r_k],
                                     trainable=True, dtype=tf.float32, initializer=Zeros)
        dists = param_dists  # (rule_n, r_k, 2)
        bias = param_bias  # (r_k, 2)
    else:
        param_dists = [tf.get_variable(name='param_dist_{}'.format(depth), shape=[rule_count, r_k, 2],
                                       trainable=True, dtype=tf.float32, initializer=tf.initializers.random_uniform)
                       for depth in range(rule_depth)]
        param_bias = tf.get_variable(name='bias', shape=[r_k, 2],
                                     trainable=True, dtype=tf.float32, initializer=tf.initializers.random_normal)
        dists = [tf.exp(param_dists[i]) for i in range(rule_depth)]  # (rule_n, r_k, 2)
        bias = tf.exp(param_bias)  # (r_k, 2)

    for i in range(rule_depth):
        tf.summary.histogram('alphas_{}'.format(i), param_alphas[i])
        tf.summary.histogram('dist_{}'.format(i), dists[i])
    tf.summary.histogram('bias', bias)

    # Features
    feats = calc_features(walks=walks,
                          assignments=walk_assignments,
                          alphas=alphas,
                          dists=dists,
                          queries=walk_queries,
                          rule_depth=rule_depth,
                          sample_count=sample_count)  # (sample_count, 2)
    ybias = tf.gather(bias, sample_queries, axis=0)
    ydist = feats + ybias
    return ydist


def cross_entropy(labels, p):
    return -tf.reduce_mean((labels * tf.log(EPSILON + p)) + ((1. - labels) * tf.log(EPSILON + 1. - p)))


def model_fn(features, labels, mode, params):
    # Loss
    logits = inference_fn(features, mode, params)
    if tf.flags.FLAGS.sigmoid_dist:
        classes = tf.greater(logits, 0)
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=logits))
    else:
        expected_p = logits[:, 0] / (EPSILON + tf.reduce_sum(logits, axis=1))
        classes = tf.argmax(input=logits, axis=1, name='classes')
        loss = cross_entropy(labels=labels, p=expected_p)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classes": classes
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode == tf.estimator.ModeKeys.TRAIN:
            eq = tf.equal(tf.cast(classes, tf.int32), tf.cast(labels, tf.int32))
            train_accuracy = tf.reduce_mean(tf.cast(eq, tf.float32))
            tf.summary.scalar('train_accuracy', train_accuracy)
            lr = tf.train.exponential_decay(params.lr,
                                            decay_rate=params.decay_rate,
                                            decay_steps=params.decay_steps,
                                            global_step=tf.train.get_global_step(),
                                            name='learning_rate',
                                            staircase=False)
            tf.summary.scalar('learning_rate', lr)
            if params.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            elif params.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=params.momentum)
            elif params.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=params.momentum)
            else:
                raise ValueError("Unknown optimizer: {}".format(params.optimizer))
            print("Trainable: {}".format(list(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))))
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        else:
            # Add evaluation metrics (for EVAL mode)
            accuracy = tf.metrics.accuracy(labels=labels, predictions=classes, name='accuracy')
            eval_metric_ops = {"accuracy": accuracy}
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def make_input_fn(rule_depth):
    def input_fn():
        pwalks = [tf.placeholder(name='walks_{}'.format(i), shape=[None, i + 1], dtype=tf.int32)
                  for i in range(rule_depth)]
        passignments = [tf.placeholder(name='assignments_{}'.format(i), shape=[None], dtype=tf.int32)
                        for i in range(rule_depth)]
        ptargets = tf.placeholder(name='targets', shape=[None], dtype=tf.int32)
        pqueries = tf.placeholder(name='queries', shape=[None], dtype=tf.int32)
        # input_fn
        kw = {'queries': pqueries}
        for i in range(rule_depth):
            kw['walks_{}'.format(i)] = pwalks[i]
            kw['assignments_{}'.format(i)] = passignments[i]
        return kw, ptargets

    return input_fn


class FeedFnHook(SessionRunHook):
    def __init__(self, path, batch_size, rule_depth):
        self.path = path
        self.batch_size = batch_size
        self.rule_depth = rule_depth
        walks = np.load(os.path.join(path, 'walks.npz'))
        self.pos_walks = [walks['pos_walks_{}'.format(i)] for i in range(rule_depth)]
        self.neg_walks = [walks['neg_walks_{}'.format(i)] for i in range(rule_depth)]
        with open(os.path.join(path, 'walk_idx.pickle'), 'rb') as f:
            self.walk_ids = pickle.load(f)
        self.count = len(self.walk_ids)

    def load_placeholders(self, graph):
        placeholder_targets = graph.get_tensor_by_name("targets:0")
        placeholder_queries = graph.get_tensor_by_name("queries:0")
        placeholder_walks = [graph.get_tensor_by_name("walks_{}:0".format(i))
                             for i in range(self.rule_depth)]
        placeholder_assignments = [graph.get_tensor_by_name("assignments_{}:0".format(i))
                                   for i in range(self.rule_depth)]
        return placeholder_walks, placeholder_queries, placeholder_targets, placeholder_assignments

    def before_run(self, run_context):
        placeholder_walks, placeholder_queries, placeholder_targets, placeholder_assignments = \
            self.load_placeholders(run_context.session.graph)
        bwalks = [[] for _ in range(self.rule_depth)]
        bassignments = [[] for _ in range(self.rule_depth)]
        btargets = []
        bqueries = []

        ids = np.random.random_integers(low=0, high=self.count - 1, size=(self.batch_size,))
        for i, id in enumerate(ids):
            walk = self.walk_ids[id]
            for d, bwalkslice in enumerate(walk['ids']):
                if bwalkslice is not None:
                    bwalk = self.pos_walks[d][bwalkslice, :]
                    bwalks[d].append(bwalk)
                    bassignments[d].append(2 * i * np.ones((bwalk.shape[0],), dtype=np.int32))
            btargets.append(1)
            bqueries.append(walk['r'])
            # print("Pos: {}".format(bwalk))
            negid = np.random.random_integers(low=0, high=len(walk['neg_ids']) - 1)
            for d, bwalkslice in enumerate(walk['neg_ids'][negid]):
                if bwalkslice is not None:
                    bwalk = self.neg_walks[d][bwalkslice, :]
                    bwalks[d].append(bwalk)
                    bassignments[d].append((2 * i * np.ones((bwalk.shape[0],), dtype=np.int32)) + 1)
            btargets.append(0)
            bqueries.append(walk['r'])
            # print("Neg: {}".format(bwalk))
            # print("R: {}".format(walk['r']))
            # raise ValueError()
        bwalks = [np.concatenate(bwalk, axis=0) for bwalk in bwalks]
        bassignments = [np.concatenate(bassignment, axis=0) for bassignment in bassignments]
        btargets = np.array(btargets, dtype=np.int32)
        bqueries = np.array(bqueries, dtype=np.int32)
        feed_dict = {placeholder_queries: bqueries, placeholder_targets: btargets}
        for pw, bw in zip(placeholder_walks, bwalks):
            feed_dict[pw] = bw
        for pa, ba in zip(placeholder_assignments, bassignments):
            feed_dict[pa] = ba
        return SessionRunArgs(
            fetches=None, feed_dict=feed_dict)


def experiment_fn(run_config, hparams):
    path = tf.flags.FLAGS.train_dir
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams)
    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=make_input_fn(hparams.rule_depth),
        train_monitors=[FeedFnHook(path=path, batch_size=hparams.batch_size, rule_depth=hparams.rule_depth)],
        eval_hooks=[FeedFnHook(path=path, batch_size=hparams.batch_size, rule_depth=hparams.rule_depth)],
        eval_input_fn=make_input_fn(hparams.rule_depth))


def main(_argv):
    print("model_dir={}".format(tf.flags.FLAGS.model_dir))
    with open('../experiments/output/FB15K/dataset/relations.pickle', 'rb') as f:
        relations = pickle.load(f)
    r_k = len(relations)
    run_config = RunConfig(model_dir=tf.flags.FLAGS.model_dir)
    hparams = HParams(lr=0.001,
                      momentum=0.9,
                      rule_depth=2,
                      rule_count=256,
                      decay_rate=0.1,
                      decay_steps=300000,
                      tau0=1.,
                      tau_decay=1e-6,
                      tau_min=0.1,
                      r_k=r_k,
                      smoothing=0.,
                      optimizer='adam',
                      batch_size=64)
    hparams.parse(tf.flags.FLAGS.hparams)
    estimator = tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=tf.flags.FLAGS.schedule,
        hparams=hparams)
