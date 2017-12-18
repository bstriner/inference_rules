import numpy as np
import tensorflow as tf
from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import Features, Feature
from tensorflow.python.lib.io.tf_record import TFRecordWriter


def feature_int64(value):
    return Feature(int64_list=tf.train.Int64List(value=value.flatten()))


def feature_bytes(value):
    return Feature(bytes_list=tf.train.BytesList(value=[value]))


def feature_array(arr):
    shp = np.array(arr.shape, np.int32)
    return feature_bytes(arr.flatten().astype(np.int32).tobytes()), feature_int64(shp)


def feature_array_int64(arr):
    shp = np.array(arr.shape, np.int32)
    return feature_int64(arr.flatten()), feature_int64(shp)


def main():
    path = 'test.tfrecord'

    with TFRecordWriter(path) as writer:
        for i in range(100):
            a, ashape = feature_array(np.arange(i + 3))
            example = Example(features=Features(feature={'a': a, 'ashape': ashape}))
            writer.write(example.SerializeToString())

    with tf.Session() as sess:
        feature = {
            'a': tf.FixedLenFeature([], tf.string),
            'ashape': tf.FixedLenFeature([1], tf.int64)
        }
        filename_queue = tf.train.string_input_producer([path])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)

        a = features['a']
        ashape = features['ashape']

        val = tf.reshape(tf.decode_raw(a, tf.int32), ashape)
        #ds = tf.data.Dataset.from_tensors(val)
        #ds = ds.batch(20)

        queue = tf.RandomShuffleQueue(100, 50, [np.int32])
        enqueue_op = queue.enqueue([val])
        qr = tf.train.QueueRunner(queue, [enqueue_op] * 1)
        tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
        d = queue.dequeue()
        d.set_shape(val.get_shape())
        ds = tf.train.batch([d], batch_size=5, capacity=100,
                       dynamic_pad=True, name='asd')
        # ds = ds.shuffle(40)
        # ds = ds.padded_batch(batch_size=32, padded_shapes=[None])
        # iterator = ds.make_initializable_iterator()
        # iterator = ds.make_one_shot_iterator()
        # tf.train.shuffle_batch()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(tf.global_variables_initializer())
        #sess.run(iterator.initializer)
        #next_element = iterator.get_next()

        for i in range(2):
            ret = sess.run(ds)
            #ret = sess.run(val)
            print("ret: {}".format(ret))
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
