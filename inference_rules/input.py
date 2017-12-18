import tensorflow as tf


def decode_array(a, ashape):
    return tf.reshape(tf.decode_raw(a, tf.int32), ashape)


def make_input_fn(data_path):
    capacity = 500
    min_after_dequeue = 200
    threads = 2
    def input_fn():
        feature = {
            's': tf.FixedLenFeature([], tf.int64),
            'r': tf.FixedLenFeature([], tf.int64),
            't': tf.FixedLenFeature([], tf.string),
            'w1': tf.FixedLenFeature([], tf.string),
            'w1shape': tf.FixedLenFeature([3], tf.int64),
            'w2': tf.FixedLenFeature([], tf.string),
            'w2shape': tf.FixedLenFeature([3], tf.int64),
            'ff': tf.FixedLenFeature([], tf.string),
            'ffshape': tf.FixedLenFeature([3], tf.int64),
            'fb': tf.FixedLenFeature([], tf.string),
            'fbshape': tf.FixedLenFeature([3], tf.int64),
        }
        filename_queue = tf.train.string_input_producer([data_path])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)

        s = tf.cast(features['s'], tf.int32)
        r = tf.cast(features['r'], tf.int32)
        t = tf.decode_raw(features['t'], tf.int32)
        w1 = decode_array(features['w1'], features['w1shape'])
        w2 = decode_array(features['w2'], features['w2shape'])
        ff = decode_array(features['ff'], features['ffshape'])
        fb = decode_array(features['fb'], features['fbshape'])

        data_single = [s, r, t, w1, w2, ff, fb]

        queue = tf.RandomShuffleQueue(capacity=capacity, min_after_dequeue=min_after_dequeue, dtypes=[tf.int32] * 7)
        enqueue_op = queue.enqueue([s, r, t, w1, w2, ff, fb])
        qr = tf.train.QueueRunner(queue, [enqueue_op] * threads)
        tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
        # tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, )
        data_shuffled = queue.dequeue()
        for a, b in zip(data_shuffled, data_single):
            a.set_shape(b.get_shape())
        data_batch = tf.train.batch(
            data_shuffled,
            batch_size=tf.flags.FLAGS.batch_size,
            capacity=capacity,
            dynamic_pad=True,
            name='shuffled_batch')

        # iterator = ds.make_initializable_iterator()
        # iterator = ds.make_one_shot_iterator()
        # tf.train.shuffle_batch()
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        # sess.run(tf.global_variables_initializer())
        # sess.run(iterator.initializer)
        # next_element = iterator.get_next()

        s, r, t, w1, w2, ff, fb = data_batch
        kw = {
            's': s,
            'r': r,
            't': t,
            'w1': w1,
            'w2': w2,
            'ff': ff,
            'fb': fb
        }
        return kw, None
    return input_fn


def make_predict_input_fn(data_path):
    capacity = 500
    min_after_dequeue = 200
    threads = 2
    def input_fn():
        feature = {
            's': tf.FixedLenFeature([], tf.int64),
            'r': tf.FixedLenFeature([], tf.int64),
            't': tf.FixedLenFeature([], tf.int64),
            'w1': tf.FixedLenFeature([], tf.string),
            'w1shape': tf.FixedLenFeature([2], tf.int64),
            'w2': tf.FixedLenFeature([], tf.string),
            'w2shape': tf.FixedLenFeature([2], tf.int64),
            'ff': tf.FixedLenFeature([], tf.string),
            'ffshape': tf.FixedLenFeature([2], tf.int64),
            'fb': tf.FixedLenFeature([], tf.string),
            'fbshape': tf.FixedLenFeature([2], tf.int64),
        }
        filename_queue = tf.train.string_input_producer([data_path])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)

        s = tf.cast(features['s'], tf.int32)
        r = tf.cast(features['r'], tf.int32)
        t = tf.cast(features['t'], tf.int32)
        w1 = decode_array(features['w1'], features['w1shape'])
        w2 = decode_array(features['w2'], features['w2shape'])
        ff = decode_array(features['ff'], features['ffshape'])
        fb = decode_array(features['fb'], features['fbshape'])

        data_single = [s, r, t, w1, w2, ff, fb]

        queue = tf.RandomShuffleQueue(capacity=capacity, min_after_dequeue=min_after_dequeue, dtypes=[tf.int32] * 7)
        enqueue_op = queue.enqueue([s, r, t, w1, w2, ff, fb])
        qr = tf.train.QueueRunner(queue, [enqueue_op] * threads)
        tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
        # tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, )
        data_shuffled = queue.dequeue()
        for a, b in zip(data_shuffled, data_single):
            a.set_shape(b.get_shape())
        data_batch = tf.train.batch(
            data_shuffled,
            batch_size=tf.flags.FLAGS.batch_size,
            capacity=capacity,
            dynamic_pad=True,
            name='shuffled_batch')

        # iterator = ds.make_initializable_iterator()
        # iterator = ds.make_one_shot_iterator()
        # tf.train.shuffle_batch()
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        # sess.run(tf.global_variables_initializer())
        # sess.run(iterator.initializer)
        # next_element = iterator.get_next()

        s, r, t, w1, w2, ff, fb = data_batch
        kw = {
            's': s,
            'r': r,
            't': t,
            'w1': w1,
            'w2': w2,
            'ff': ff,
            'fb': fb
        }
        return kw, None
    return input_fn