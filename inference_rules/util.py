import tensorflow as tf

EPSILON = 1e-7


def masked_feature(raw):
    feat = tf.maximum(raw - 1, 0)
    mask = tf.cast(tf.not_equal(raw, 0), tf.float32)
    return feat, mask


def softmax_nd(x, axis=-1):
    e_x = tf.exp(x - tf.reduce_max(x, axis=axis, keep_dims=True))
    out = e_x / tf.reduce_sum(e_x, axis=axis, keep_dims=True)
    return out


def softmax_nd_masked(x, mask, axis=-1):
    e_x = tf.exp(x - tf.reduce_max(x, axis=axis, keep_dims=True))
    e_x *= mask
    out = e_x / tf.reduce_sum(e_x, axis=axis, keep_dims=True)
    return out
