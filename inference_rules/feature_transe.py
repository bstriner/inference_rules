import tensorflow as tf


def feature_transe(s, r, t, params):
    r_ctx = tf.layers.dense(
        inputs=r,
        units=params.units,
        name='transe_ctx_r')  # (n, units)
    edense = tf.layers.Dense(
        units=params.units,
        name='transe_ctx_e')
    s_ctx = edense(s)
    t_ctx = edense(t)
    score = -tf.reduce_sum(tf.square(s_ctx + r_ctx - t_ctx), 1)
    return score
