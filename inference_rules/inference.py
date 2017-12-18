import tensorflow as tf

from .feature_secondary import feature_secondary_masked
from .feature_transe import feature_transe
from .feature_walks import feature_walks_masked
from .util import masked_feature


def inference_fn_flat(s, r, t, w1_raw, w2_raw, ff_raw, fb_raw, e_k, r_k, params, mode):
    """

    :param s:
    :param r:
    :param t:
    :param w1_raw:
    :param w2_raw:
    :param ff_raw: (n*samples, walks, 2)
    :param fb_raw: (n*samples, walks, 2)
    :param e_k:
    :param r_k:
    :param params:
    :param mode:
    :return:
    """
    initializer = tf.initializers.random_uniform(-0.05, 0.05)

    entity_embedding = tf.get_variable(
        name='entity_embedding',
        shape=[e_k, params.units],
        trainable=True,
        dtype=tf.float32,
        initializer=initializer)
    relation_embedding = tf.get_variable(
        name='relation_embedding',
        shape=[r_k * 2, params.units],
        trainable=True,
        dtype=tf.float32,
        initializer=initializer)

    s_embedded = tf.nn.embedding_lookup(entity_embedding, s)  # (n, units)
    r_embedded = tf.nn.embedding_lookup(relation_embedding, r)  # (n, units)
    t_embedded = tf.nn.embedding_lookup(entity_embedding, t)  # (n, units)

    scores = 0

    if params.enable_transe:
        feat_transe = feature_transe(s_embedded, r_embedded, t_embedded, params=params)  # (n,)
        scores += feat_transe

    if params.enable_walks1:
        w1, w1mask = masked_feature(w1_raw)
        w1_embedded = tf.nn.embedding_lookup(relation_embedding, w1)  # (n, walks, 1, units)
        feat_walks1 = feature_walks_masked(
            r_embedded=r_embedded,
            walks_embedded=w1_embedded,
            walk_mask=w1mask,
            params=params,
            mode=mode)
        print("feat_walks1: {}".format(feat_walks1))
        scores += feat_walks1

    if params.enable_walks2:
        w2, w2mask = masked_feature(w2_raw)
        w2_embedded = tf.nn.embedding_lookup(relation_embedding, w2)  # (n, walks, 2, units)
        feat_walks2 = feature_walks_masked(
            r_embedded=r_embedded,
            walks_embedded=w2_embedded,
            walk_mask=w2mask,
            params=params,
            mode=mode)
        print("feat_walks2: {}".format(feat_walks2))
        scores += feat_walks2

    if params.enable_secondary:
        ff, ffmask = masked_feature(ff_raw)
        fent = tf.nn.embedding_lookup(entity_embedding, ff[:, :, 0])  # (n, walks, units)
        frel = tf.nn.embedding_lookup(relation_embedding, ff[:, :, 1])  # (n, walks, units)
        fscore = feature_secondary_masked(
            secondary_ent_embedded=fent,
            rel_embedded=frel,
            target_embedded=t_embedded,
            query_embedded=r_embedded,
            mask=ffmask,
            params=params
        )

        fb, fbmask = masked_feature(fb_raw)
        bent = tf.nn.embedding_lookup(entity_embedding, fb[:, :, 0])  # (n, walks, units)
        brel = tf.nn.embedding_lookup(relation_embedding, fb[:, :, 1])  # (n, walks, units)
        r_reversed_embedded = tf.nn.embedding_lookup(relation_embedding, r + r_k)  # (n, units)
        bscore = feature_secondary_masked(
            secondary_ent_embedded=bent,
            rel_embedded=brel,
            target_embedded=s_embedded,
            query_embedded=r_reversed_embedded,
            mask=fbmask,
            params=params
        )
        scores += fscore + bscore

    return scores


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
        params=params,
        mode=mode
    )
    return scores


def inference_fn(features, mode, params, e_k, r_k):
    s = features['s']  # (n,)
    r = features['r']  # (n,)
    t_raw = features['t']  # (n, samples)
    w1_raw = features['w1']  # (n, samples, walks, 1)
    w2_raw = features['w2']  # (n, samples, walks, 2)
    ff_raw = features['ff']  # (n, samples, walks, 2)
    fb_raw = features['fb']  # (n, samples, walks, 2)

    n = tf.shape(s)[0]
    samples = tf.shape(t_raw)[1]

    t, t_mask = masked_feature(t_raw)
    srep = tf.tile(tf.expand_dims(s, 1), [1, samples])
    rrep = tf.tile(tf.expand_dims(r, 1), [1, samples])

    sflat = tf.reshape(srep, (-1,))  # (n*samples,)
    rflat = tf.reshape(rrep, (-1,))  # (n*samples,)
    tflat = tf.reshape(t, (-1,))  # (n*samples,)
    w1flat = tf.reshape(w1_raw, (-1, tf.shape(w1_raw)[2], 1))  # (n*samples, walks, 1)
    w2flat = tf.reshape(w2_raw, (-1, tf.shape(w2_raw)[2], 2))  # (n*samples, walks, 2)
    ffflat = tf.reshape(ff_raw, (-1, tf.shape(ff_raw)[2], 2))
    fbflat = tf.reshape(fb_raw, (-1, tf.shape(fb_raw)[2], 2))

    scores_flat = inference_fn_flat(
        s=sflat,
        r=rflat,
        t=tflat,
        e_k=e_k,
        r_k=r_k,
        w1_raw=w1flat,
        w2_raw=w2flat,
        ff_raw=ffflat,
        fb_raw=fbflat,
        params=params,
        mode=mode
    )

    scores = tf.reshape(scores_flat, tf.shape(t_raw))
    scores *= t_mask
    scores -= (1. - t_mask) * 1e6
    return scores, t_mask
