import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
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

def generate_rules(walk_queries, r_k, rule_count, units, temperature):
    activation = tf.nn.relu
    rule_depth = len(walk_queries)
    rules = []
    for i, query in enumerate(walk_queries):
        embedding_r = tf.get_variable(name='embedding_r', shape=[r_k, units], trainable=True)
        embedding_k = tf.get_variable(name='embedding_k', shape=[rule_count, units], trainable=True)
        embedding_depth = tf.get_variable(name='embedding_d', shape=[rule_depth, units], trainable=True)
        embedded_r = tf.gather(embedding_r, query, axis=0)  # (wn, units)
        embedded_d = tf.gather(embedding_depth, i, axis=0) # (units,)
        ctx = tf.expand_dims(embedded_r, 1) + tf.expand_dims(embedding_k, 0) + embedded_d  # (wn, rn, units)
        h0 = tf.get_variable(name='rnn_h0', shape=[units], trainable=True)
        alphas = []
        for j in range(i+1):
            h = h0 + ctx
            h = activation(h)
            h = tf.layers.dense(h, units=units, activation=activation, name='h1_dense_1')
            h0 = tf.layers.dense(h, units=units, name='h1_dense 2')
            h = tf.layers.dense(h, units=units, activation=activation, name='alpha_dense_1')
            h = tf.layers.dense(h, units=r_k*2, name='alpha_dense_2')
            alpha = gumbel_sigmoid(h, temperature=temperature) # (wn, rn, rk*2)
            alphas.append(alpha)
        h = activation(ctx)
        h = tf.layers.dense(h, units=units, activation=activation, name='dist_dense_1')
        h = tf.layers.dense(h, units=1, name='dist_dense_2')
        dist = tf.squeeze(h, axis=-1) # (wn, rn)
        if i > 0:
            alphas = tf.stack(alphas, axis=2) # (wn, rn, d, rk*2)
        else:
            alphas = tf.expand_dims(alphas[0], axis=2)
        rules.append({'alphas':alphas, 'dist':dist})
    return rules

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

    rules = generate_rules(walk_queries=walk_queries,r_k=r_k,
                           rule_count=rule_count, units=params.units, temperature=tau)

    for walk, assignment, query, depth, rule in zip(walks, walk_assignments, walk_queries, range(rule_depth), rules):
        alpha = rule['alphas']# (wn, rn, d, rk*2)
        dist = rule['dist']# (wn, rn)
        # walk: (rn, d)




    param_alphas = [tf.get_variable(name='alphas_{}'.format(depth), shape=[rule_count, depth + 1, r_k * 2],
                                    trainable=True, dtype=tf.float32, initializer=tf.initializers.random_normal)
                    for depth in range(rule_depth)]
    alphas = [gumbel_sigmoid(param_alphas[i], temperature=tau)  # (rd, rn, rk*2)
              for i in range(rule_depth)]

    if tf.flags.FLAGS.sigmoid_dist:
        param_dists = [tf.get_variable(name='param_dist_{}'.format(depth), shape=[rule_count, r_k],
                                       trainable=True, dtype=tf.float32, initializer=tf.initializers.random_normal)
                       for depth in range(rule_depth)]
        param_bias = tf.get_variable(name='bias', shape=[r_k],
                                     trainable=True, dtype=tf.float32, initializer=tf.initializers.random_normal)
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
        self.placeholder_targets = None
        self.placeholder_queries = None
        self.placeholder_walks = None
        self.placeholder_assignments = None
        self.loaded = False
        walks = np.load(os.path.join(path, 'walks.npz'))
        self.pos_walks = [walks['pos_walks_{}'.format(i)] for i in range(rule_depth)]
        self.neg_walks = [walks['neg_walks_{}'.format(i)] for i in range(rule_depth)]
        with open(os.path.join(path, 'walk_idx.pickle'), 'rb') as f:
            self.walk_ids = pickle.load(f)
        self.count = len(self.walk_ids)

    def load_placeholders(self, graph):
        self.placeholder_targets = graph.get_tensor_by_name("targets:0")
        self.placeholder_queries = graph.get_tensor_by_name("queries:0")
        self.placeholder_walks = [graph.get_tensor_by_name("walks_{}:0".format(i))
                                  for i in range(self.rule_depth)]
        self.placeholder_assignments = [graph.get_tensor_by_name("assignments_{}:0".format(i))
                                        for i in range(self.rule_depth)]
        self.loaded = True

    def before_run(self, run_context):
        # if not self.loaded:
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
        feed_dict = {self.placeholder_queries: bqueries, self.placeholder_targets: btargets}
        for pw, bw in zip(self.placeholder_walks, bwalks):
            feed_dict[pw] = bw
        for pa, ba in zip(self.placeholder_assignments, bassignments):
            feed_dict[pa] = ba
        return SessionRunArgs(
            fetches=None, feed_dict=feed_dict)


class LogPredictionsHook(SessionRunHook):
    def __init__(self, every_n_iter=None, every_n_secs=None):
        if every_n_iter is not None and every_n_iter <= 0:
            raise ValueError("invalid every_n_iter=%s." % every_n_iter)
        self._timer = (
            SecondOrStepTimer(every_secs=every_n_secs, every_steps=every_n_iter))

    def begin(self):
        self._timer.reset()
        self._iter_count = 0

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        if self._should_trigger:
            return SessionRunArgs(self._current_tensors)
        else:
            return None

    def after_run(self, run_context, run_values):
        self._iter_count += 1


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
    run_config = tf.contrib.learn.RunConfig(model_dir=tf.flags.FLAGS.model_dir)
    hparams = tf.contrib.training.HParams(lr=0.001,
                                          momentum=0.9,
                                          rule_depth=2,
                                          rule_count=256,
                                          decay_rate=0.1,
                                          decay_steps=300000,
                                          tau0=5.,
                                          tau_decay=1e-6,
                                          tau_min=0.1,
                                          r_k=r_k,
                                          smoothing=0.,
                                          optimizer='adam',
                                          batch_size=256)
    hparams.parse(tf.flags.FLAGS.hparams)
    estimator = tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=tf.flags.FLAGS.schedule,
        hparams=hparams)
