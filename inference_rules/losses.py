import tensorflow as tf

from .util import EPSILON


def cross_entropy_loss(labels, p, params, mask):
    depth = tf.reduce_sum(mask, axis=-1, keep_dims=True)  # (n,)
    if params.smoothing > 0:
        labels = (labels * (1. - params.smoothing)) + (params.smoothing * tf.ones_like(labels) / depth)
    loss_partial = -(labels * tf.log(EPSILON + p)) - ((1. - labels) * tf.log(EPSILON + 1. - p))  # (n, samples)
    loss_partial *= mask
    loss_partial = tf.reduce_sum(loss_partial, axis=1)  # (n,)
    #loss_partial /= tf.squeeze(depth, 1)
    return loss_partial
