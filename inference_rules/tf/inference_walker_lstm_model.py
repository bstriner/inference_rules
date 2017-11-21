import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs

EPSILON = 1e-7
from tensorflow.contrib.rnn import LSTMCell

from tensorflow.python.ops.init_ops import RandomUniform


def run_lstm(walks, walk_assignments, walk_queries, units, rule_depth, r_k, sample_queries, sample_count,
             regularizer=None,
             layers=2):
    betas = 0.
    lstms = []
    initializer = RandomUniform(minval=-0.05, maxval=0.05)
    for i in range(layers):
        lstms.append(LSTMCell(units, initializer=initializer))
    embedding_depth = tf.get_variable(name='embedding_d', shape=[rule_depth, units], trainable=True,
                                      initializer=initializer)
    embedding_q = tf.get_variable(name='embedding_q', shape=[r_k, units], trainable=True, initializer=initializer)
    embedding_r = tf.get_variable(name='embedding_r', shape=[r_k * 2, units], trainable=True, initializer=initializer)
    # beta_dense = tf.layers.Dense(units=2, kernel_initializer=initializer, name='beta_dense')

    for walk, assignment, query, depth in zip(walks, walk_assignments, walk_queries, range(rule_depth)):
        # walks: (wn, rd)
        # activation = tf.nn.relu
        embedded_depth = tf.gather(embedding_depth, depth, axis=0)
        embedded_q = tf.gather(embedding_q, query, axis=0)
        batch_size = tf.shape(assignment)[0]
        # Initial state of the LSTM memory.
        states = [lstm.zero_state(batch_size=batch_size, dtype=tf.float32) for lstm in lstms]
        for w in tf.unstack(walk, axis=1):
            embedded_w = tf.gather(embedding_r, w, axis=0)
            out = embedded_w + embedded_depth + embedded_q
            for i, lstm in enumerate(lstms):
                with tf.variable_scope('lstm_{}'.format(i), reuse=tf.AUTO_REUSE):
                    out, state = lstm(out, states[i])
                states[i] = state
        # out = tf.layers.dense(out, units=units, name='beta_hidden', reuse=tf.AUTO_REUSE, activation=tf.nn.relu,
        #                    kernel_initializer=initializer)
        # h = tf.layers.dense(out, units=2, name='beta_dense', reuse=tf.AUTO_REUSE, kernel_initializer=initializer)
        with tf.variable_scope('beta_dense_scope', reuse=tf.AUTO_REUSE):
            out = tf.layers.dense(out, units=2, kernel_initializer=initializer)
        # out = beta_dense.apply(out)
        # out = tf.layers.dense(out, units=2, name='beta_dense_{}'.format(depth), kernel_initializer=initializer)
        #beta = tf.nn.relu(out, name='beta_relu_{}'.format(depth))  # (wn, 2)
        beta = tf.exp(out, name='beta_exp_{}'.format(depth))  # (wn, 2)
        assignment_one_hot = tf.one_hot(assignment, sample_count,
                                        name='assignment_one_hot_{}'.format(depth))  # (wn, sample_count)
        betas += tf.tensordot(assignment_one_hot, beta, axes=(0, 0),
                              name='beta_assignment_{}'.format(depth))  # (sample_count, 2)

    bias_var = tf.get_variable(name='beta_bias', shape=[r_k, 2],
                               trainable=True, dtype=tf.float32, initializer=tf.initializers.zeros)
    bias = tf.exp(tf.gather(bias_var, sample_queries, axis=0))
    #bias = tf.nn.relu(tf.gather(bias_var, sample_queries, axis=0), name='bias_relu')
    betas += bias
    betas += EPSILON
    return betas


def inference_fn(features, mode, params):
    rule_depth = params.rule_depth
    r_k = params.r_k
    units = params.units

    # Inputs
    walks = [features['walks_{}'.format(i)] for i in range(rule_depth)]
    walk_assignments = [features['assignments_{}'.format(i)] for i in range(rule_depth)]
    # walk_assignments: (wn,) int [0-sample_count]
    sample_queries = features['queries']
    # targets = features['targets']
    sample_count = tf.shape(sample_queries)[0]
    walk_queries = [tf.gather(sample_queries, walk_assignments[i]) for i in range(rule_depth)]

    betas = run_lstm(walks=walks, walk_assignments=walk_assignments, walk_queries=walk_queries,
                     units=units, rule_depth=rule_depth, r_k=r_k,
                     sample_queries=sample_queries, sample_count=sample_count, layers=2)
    tf.summary.histogram('betas_0', betas[:, 0])
    tf.summary.histogram('betas_1', betas[:, 1])
    expected_p = tf.gather(betas, 0, axis=1) / tf.reduce_sum(betas, axis=1)
    tf.summary.histogram('expected_p', expected_p)
    return expected_p


def cross_entropy(labels, p):
    return -tf.reduce_mean((labels * tf.log(EPSILON + p)) + ((1. - labels) * tf.log(EPSILON + 1. - p)))


def model_fn(features, labels, mode, params):
    # Loss
    expected_p = inference_fn(features, mode, params)
    #logits = tf.log(expected_p)
    smooth_labels = tf.cast(labels, tf.float32)
    if params.smoothing > 0:
        smooth_labels = (smooth_labels * (1.-params.smoothing))+(params.smoothing/2.)
    #loss = tf.reduce_mean(
    #    tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    loss = cross_entropy(labels=smooth_labels, p=expected_p)
    classes = tf.greater(expected_p, 0.5)
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


def make_input_fn(rule_depth, evaluation=False):
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
        if evaluation:
            kw['query_groups'] = tf.placeholder(name='query_groups', shape=[None, 2], dtype=tf.int32)
        return kw, ptargets

    return input_fn

class FeedFnHook(SessionRunHook):
    def __init__(self, path, batch_size, rule_depth, evaluation=False):
        self.path = path
        self.batch_size = batch_size
        self.rule_depth = rule_depth
        self.placeholder_targets = None
        self.placeholder_queries = None
        self.placeholder_walks = None
        self.placeholder_assignments = None
        self.placeholder_query_groups = None
        self.loaded = False
        self.evaluation=evaluation
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
        if self.evaluation:
            self.placeholder_query_groups = graph.get_tensor_by_name('query_groups:0')

        self.loaded = True

    def build_feed_dict(self, run_context):
        self.load_placeholders(run_context.session.graph)
        bwalks = [[] for _ in range(self.rule_depth)]
        bassignments = [[] for _ in range(self.rule_depth)]
        btargets = []
        bqueries = []
        bquery_groups = []
        neg_count = 20 if self.evaluation else 1

        ids = np.random.random_integers(low=0, high=self.count - 1, size=(self.batch_size,))
        walk_counter = 0
        for i, id in enumerate(ids):
            walk = self.walk_ids[id]
            for d, bwalkslice in enumerate(walk['ids']):
                if bwalkslice is not None:
                    bwalk = self.pos_walks[d][bwalkslice, :]
                    bwalks[d].append(bwalk)
                    bassignments[d].append(walk_counter * np.ones((bwalk.shape[0],), dtype=np.int32))
            btargets.append(1)
            bqueries.append(walk['r'])
            if self.evaluation:
                bquery_groups.append([i, 0])
            walk_counter += 1
            # print("Pos: {}".format(bwalk))
            neg_ids = walk['neg_ids']
            neg_count_t = min([neg_count, len(neg_ids)])
            #negid = np.random.random_integers(low=0, high=len(neg_ids) - 1, size=(neg_count_t,))
            negid = np.random.choice(a=range(len(neg_ids)), replace=False, size=(neg_count_t,))
            for j, negidt in enumerate(negid):
                for d, bwalkslice in enumerate(neg_ids[negidt]):
                    if bwalkslice is not None:
                        bwalk = self.neg_walks[d][bwalkslice, :]
                        bwalks[d].append(bwalk)
                        bassignments[d].append((walk_counter * np.ones((bwalk.shape[0],), dtype=np.int32)))
                btargets.append(0)
                bqueries.append(walk['r'])
                if self.evaluation:
                    bquery_groups.append([i, 1+j])
                walk_counter += 1
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
        if self.evaluation:
            feed_dict[self.placeholder_query_groups] = np.array(bquery_groups, dtype=np.int32)
        return feed_dict

    def before_run(self, run_context):
        # if not self.loaded:
        feed_dict = self.build_feed_dict(run_context=run_context)
        return SessionRunArgs(
            fetches=None, feed_dict=feed_dict)


"""
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
"""


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
                                          units=512,
                                          # rule_count=256,
                                          decay_rate=0.1,
                                          decay_steps=300000,
                                          sigmoid_output=False,
                                          r_k=r_k,
                                          smoothing=0.1,
                                          optimizer='adam',
                                          batch_size=64)
    hparams.parse(tf.flags.FLAGS.hparams)
    estimator = tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=tf.flags.FLAGS.schedule,
        hparams=hparams)
