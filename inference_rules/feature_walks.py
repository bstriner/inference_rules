import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell


def feature_walks_masked(
        r_embedded,
        walks_embedded,
        walk_mask,
        params,
        mode,
        layers=2,
        reuse=tf.AUTO_REUSE
):
    """

    :param r_embedded: (n*samples, units)
    :param walks_embedded:  (n*samples, walks, depth, units)
    :param walk_mask: (n*samples, walks)
    :param params:
    :param mode:
    :param layers:
    :param reuse:
    :return:
    """
    shape = tf.shape(walks_embedded)
    walk_count = shape[1]
    depth = shape[2]
    r_tiled = tf.tile(tf.expand_dims(r_embedded, 1), [1, walk_count, 1])  # (n*samples, walks, units)
    r_flat = tf.reshape(r_tiled, (-1, params.units))  # (n*samples*walks, units)
    w_flat = tf.reshape(walks_embedded, (-1, depth, params.units))  # (n*samples*walks, depth, units)
    walk_mask = tf.reduce_sum(walk_mask, 2)
    scores_flat = feature_walks(
        r_embedded=r_flat,
        walks_embedded=w_flat,
        params=params,
        mode=mode,
        layers=layers,
        reuse=reuse
    )  # (n*samples*walks,)
    scores = tf.reshape(scores_flat, (-1, walk_count))  # (n*samples, walks)
    scores *= walk_mask
    scores = tf.reduce_sum(scores, axis=1)  # (n*samples,)
    return scores


def feature_walks(
        r_embedded,
        walks_embedded,
        params,
        mode,
        layers=2,
        reuse=tf.AUTO_REUSE):
    units = params.units
    lstms = []
    for i in range(layers):
        lstms.append(LSTMCell(units))

    r_ctx = tf.layers.dense(
        inputs=r_embedded,
        units=params.units,
        name='walk_ctx_r',
        reuse=reuse)  # (n, units)
    w_ctx = tf.layers.dense(
        inputs=walks_embedded,
        units=params.units,
        name='walk_ctx_w',
        reuse=reuse)  # (n, depth, units)
    ctx = tf.expand_dims(r_ctx, 1) + w_ctx

    training = mode == tf.estimator.ModeKeys.TRAIN
    batch_size = tf.shape(r_embedded)[0]

    h = ctx
    for i, lstm in enumerate(lstms):
        with tf.variable_scope('walk_lstm_{}'.format(i), reuse=reuse):
            initial_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
            h, state = tf.nn.dynamic_rnn(
                cell=lstm,
                inputs=h,
                initial_state=initial_state,
                time_major=False)
    output = tf.layers.dense(
        inputs=h[:, -1, :],
        units=1,
        reuse=reuse,
        name='walk_score'
    )  # (n, 1)
    return tf.squeeze(output, 1)  # (n,)
