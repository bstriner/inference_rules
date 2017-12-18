import tensorflow as tf


def main():
    data_path = 'output/processed/valid.tfrecords'
    with tf.Session() as sess:
        feature = {
            's': tf.FixedLenFeature([], tf.int64),
                   'r': tf.FixedLenFeature([], tf.int64),
                   't': tf.FixedLenFeature([], tf.string),
                   'w1': tf.FixedLenFeature([], tf.string),
                   'w1shape': tf.FixedLenFeature([], tf.string),
                   'w2': tf.FixedLenFeature([], tf.string),
                   'w2shape': tf.FixedLenFeature([], tf.string),
                   'ff': tf.FixedLenFeature([], tf.string),
                   'ffshape': tf.FixedLenFeature([], tf.string),
                   'fb': tf.FixedLenFeature([], tf.string),
                   'fbshape': tf.FixedLenFeature([], tf.string),
                   }
        filename_queue = tf.train.string_input_producer([data_path])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers

        s = features['s']
        r = features['r']
        t = tf.decode_raw(features['t'], tf.int32)
        w1 = tf.reshape(tf.decode_raw(features['w1'], tf.int32), tf.decode_raw(features['w1shape'], tf.int32))
        w2 = tf.reshape(tf.decode_raw(features['w2'], tf.int32), tf.decode_raw(features['w2shape'], tf.int32))
        ff = tf.reshape(tf.decode_raw(features['ff'], tf.int32), tf.decode_raw(features['ffshape'], tf.int32))
        fb = tf.reshape(tf.decode_raw(features['fb'], tf.int32), tf.decode_raw(features['fbshape'], tf.int32))

        #batch = tf.data.Dataset.pa

        ds = tf.data.Dataset.from_tensors((s,r,t,w1,w2,ff,fb))
        #ds = tf.data.Dataset.zip((s, r, t, w1, w2, ff, fb))
        ds = ds.shuffle(256, reshuffle_each_iteration=True)
        ds = ds.padded_batch(
            batch_size=32,
            padded_shapes=(
                [],
                [],
                [None],
                [None, None, None],
                [None, None, None],
                [None, None, None],
                [None, None, None])
        )
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(coord=coord)
        iterator = ds.make_initializable_iterator()
        sess.run(iterator.initializer)
        next_element = iterator.get_next()


        for i in range(20):
            print("Data")
            #ret = sess.run([s, r, t, w1, w2, ff, fb])
            ret = sess.run(next_element)
            for x in ret:
                print(x.shape)

        coord.request_stop()
        coord.join(threads)

    pass


if __name__ == '__main__':
    main()
