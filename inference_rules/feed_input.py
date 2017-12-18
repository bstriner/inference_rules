import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs

from .dataset import DatasetProcessor


def make_input_fn(params):
    def input_fn():
        kw = {
            's': tf.placeholder(tf.int32, [None], name='s'),
            'r': tf.placeholder(tf.int32, [None], name='r'),
            't': tf.placeholder(tf.int32, [None], name='t'),
            'tneg': tf.placeholder(tf.int32, [None], name='tneg'),
            'tnegassignments': tf.placeholder(tf.int32, [None, 2], name='tnegassignments'),
        }
        if params.enable_walks1:
            kw['w1s'] = tf.placeholder(tf.int32, [None, 1], name='w1s')
            kw['w1assignments'] = tf.placeholder(tf.int32, [None, 2], name='w1assignments')
        if params.enable_walks2:
            kw['w2s'] = tf.placeholder(tf.int32, [None, 2], name='w2s')
            kw['w2assignments'] = tf.placeholder(tf.int32, [None, 2], name='w2assignments')
        if params.enable_secondary:
            kw['ffs'] = tf.placeholder(tf.int32, [None, 2], name='ffs')
            kw['ffassignments'] = tf.placeholder(tf.int32, [None, 2], name='ffassignments')
            kw['fbs'] = tf.placeholder(tf.int32, [None, 2], name='fbs')
            kw['fbassignments'] = tf.placeholder(tf.int32, [None, 2], name='fbassignments')
        return kw, None

    return input_fn


class FeedFnHook(SessionRunHook):
    def __init__(self, input_path, mode, batch_size, params, negative_samples=200):
        self.dataset = DatasetProcessor(
            input_path=input_path,
            mode=mode,
            negative_samples=negative_samples,
            params=params)
        self.batch_size = batch_size
        self.batch_iter = self.gen_splits_forever()
        self.params = params
        self.placeholders = {}

    def load_placeholders(self, graph):
        self.placeholders = {
            's': graph.get_tensor_by_name("s:0"),
            'r': graph.get_tensor_by_name("r:0"),
            't': graph.get_tensor_by_name("t:0"),
            'tneg': graph.get_tensor_by_name("tneg:0"),
            'tnegassignments': graph.get_tensor_by_name("tnegassignments:0"),
        }

        if self.params.enable_walks1:
            self.placeholders['w1s'] = graph.get_tensor_by_name("w1s:0")
            self.placeholders['w1assignments'] = graph.get_tensor_by_name("w1assignments:0")
        if self.params.enable_walks2:
            self.placeholders['w2s'] = graph.get_tensor_by_name("w2s:0")
            self.placeholders['w2assignments'] = graph.get_tensor_by_name("w2assignments:0")
        if self.params.enable_secondary:
            self.placeholders['ffs'] = graph.get_tensor_by_name("ffs:0")
            self.placeholders['ffassignments'] = graph.get_tensor_by_name("ffassignments:0")
            self.placeholders['fbs'] = graph.get_tensor_by_name("fbs:0")
            self.placeholders['fbassignments'] = graph.get_tensor_by_name("fbassignments:0")

    def after_create_session(self, session, coord):
        self.load_placeholders(session.graph)

    def generate_batch(self):
        data = self.dataset.generate_examples(batch_size=self.batch_size)
        kw = {}
        for field in ['s', 'r', 't', 'tneg', 'tnegassignments']:
            kw[self.placeholders[field]] = data[field]
        if self.params.enable_walks1:
            kw[self.placeholders['w1s']] = data['w1s']
            kw[self.placeholders['w1assignments']] = data['w1assignments']
        if self.params.enable_walks2:
            kw[self.placeholders['w2s']] = data['w2s']
            kw[self.placeholders['w2assignments']] = data['w2assignments']
        if self.params.enable_secondary:
            kw[self.placeholders['ffs']] = data['ffs']
            kw[self.placeholders['ffassignments']] = data['ffassignments']
            kw[self.placeholders['fbs']] = data['fbs']
            kw[self.placeholders['fbassignments']] = data['fbassignments']
        return kw

    def gen_splits_forever(self):
        while True:
            yield self.generate_batch()

    def before_run(self, run_context):
        feed_dict = next(self.batch_iter)
        return SessionRunArgs(fetches=None, feed_dict=feed_dict)
