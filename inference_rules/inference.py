import tensorflow as tf

from .util import masked_feature


def feature_transe(s, r, t):
    score = -tf.reduce_sum(tf.square(s + r - t), 1)
    return score


def inference_fn_flat(s, r, t, e_k, r_k, params):
    initializer = tf.initializers.random_uniform(-0.05, 0.05)

    entity_embedding = tf.get_variable(
        name='entity_embedding',
        shape=[e_k, params.units],
        trainable=True,
        dtype=tf.float32,
        initializer=initializer)
    relation_embedding = tf.get_variable(
        name='relation_embedding',
        shape=[r_k, params.units],
        trainable=True,
        dtype=tf.float32,
        initializer=initializer)

    s_embedded = tf.nn.embedding_lookup(entity_embedding, s)  # (n, units)
    r_embedded = tf.nn.embedding_lookup(relation_embedding, r)  # (n, units)
    t_embedded = tf.nn.embedding_lookup(entity_embedding, t)  # (n, units)

    feat_transe = feature_transe(s_embedded, r_embedded, t_embedded)  # (n,)

    total_features = feat_transe
    return total_features


def inference_fn_pred(features, mode, params, e_k, r_k):
    s = features['s']
    r = features['r']
    t = features['t']
    scores = inference_fn_flat(
        s=s,
        r=r,
        t=t,
        e_k=e_k,
        r_k=r_k,
        params=params
    )
    return scores


def inference_fn(features, mode, params, e_k, r_k):
    s = features['s']  # (n,)
    r = features['r']  # (n,)
    t_raw = features['t']  # (n, samples)

    n = tf.shape(s)[0]
    samples = tf.shape(t_raw)[1]

    t, t_mask = masked_feature(t_raw)
    srep = tf.tile(tf.expand_dims(s, 1), [1, samples])
    rrep = tf.tile(tf.expand_dims(r, 1), [1, samples])

    sflat = tf.reshape(srep, (-1,))
    rflat = tf.reshape(rrep, (-1,))
    tflat = tf.reshape(t, (-1,))

    scores_flat = inference_fn_flat(
        s=sflat,
        r=rflat,
        t=tflat,
        e_k=e_k,
        r_k=r_k,
        params=params)

    scores = tf.reshape(scores_flat, tf.shape(t_raw))
    scores *= t_mask
    scores -= (1. - t_mask) * 1e6
    return scores, t_mask
