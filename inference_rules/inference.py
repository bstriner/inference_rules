import tensorflow as tf

from .feature_embedding import GraphEmbedding
from .feature_secondary import feature_secondary
from .feature_transe import feature_transe
from .feature_walks import feature_walks
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
    embedding = GraphEmbedding(e_k=e_k, r_k=r_k, params=params, training=mode == tf.estimator.ModeKeys.TRAIN)
    s = features['s']  # (n,)
    r = features['r']  # (n,)
    t = features['t']  # (n,)
    tneg = features['tneg']  # (n,)
    tnegassignments = features['tnegassignments']  # (n, samples)
    maxneg = tf.reduce_max(tnegassignments[:, 1], 0)  # scalar
    n = tf.shape(s)[0]

    scores = tf.zeros((n, maxneg + 1), dtype=tf.float32)
    print("shape: {}".format(tf.shape(scores)))
    score_mask = tf.scatter_nd(
        indices=tnegassignments,
        updates=tf.ones((tf.shape(tnegassignments)[0],), tf.float32),
        shape=tf.shape(scores))
    score_mask = tf.concat((tf.ones((n, 1)), score_mask[:, 1:]), axis=1)

    if params.enable_transe:
        h = feature_transe(s, r, t, embedding=embedding, params=params)
        scores += tf.concat((tf.expand_dims(h, 1), tf.zeros((n, maxneg), dtype=tf.float32)), axis=1)

        sneg = tf.gather(s, tnegassignments[:, 0], axis=0)
        rneg = tf.gather(r, tnegassignments[:, 0], axis=0)
        h = feature_transe(sneg, rneg, tneg, embedding=embedding, params=params)
        scores += tf.scatter_nd(indices=tnegassignments, updates=h, shape=tf.shape(scores))

    if params.enable_walks1:
        w1s = features['w1s']
        w1assignments = features['w1assignments']
        rw1 = tf.gather(r, w1assignments[:, 0], axis=0)
        h = feature_walks(
            r=rw1,
            w=w1s,
            embedding=embedding,
            params=params,
            mode=mode)
        scores += tf.scatter_nd(indices=w1assignments, updates=h, shape=tf.shape(scores))

    if params.enable_walks2:
        w2s = features['w2s']
        w2assignments = features['w2assignments']
        rw2 = tf.gather(r, w2assignments[:, 0], axis=0)
        h = feature_walks(
            r=rw2,
            w=w2s,
            embedding=embedding,
            params=params,
            mode=mode)
        scores += tf.scatter_nd(indices=w2assignments, updates=h, shape=tf.shape(scores))

    if params.enable_secondary:
        ffs = features['ffs']
        ffassignments = features['ffassignments']
        ts = tf.scatter_nd(indices=tnegassignments, updates=tneg, shape=tf.shape(scores))
        ts = tf.concat((tf.expand_dims(t, 1), ts[:, 1:]), axis=1)
        ft = tf.gather_nd(ts, indices=ffassignments)
        fe = ffs[:, 0]
        fr = ffs[:, 1]
        fq = tf.gather(r, ffassignments[:, 0], axis=0)

        fscores = feature_secondary(
            ent=fe,
            rel=fr,
            q=fq,
            t=ft,
            embedding=embedding,
            params=params,
        )
        scores += tf.scatter_nd(indices=ffassignments, updates=fscores, shape=tf.shape(scores))

        fbs = features['fbs']
        fbassignments = features['fbassignments']
        ss = tf.gather(s, indices=fbassignments[:, 0], axis=0)
        be = fbs[:, 0]
        br = fbs[:, 1]
        bq = tf.gather(r, fbassignments[:, 0], axis=0)

        bscores = feature_secondary(
            ent=be,
            rel=br,
            q=bq + r_k,
            t=ss,
            embedding=embedding,
            params=params,
        )
        scores += tf.scatter_nd(indices=fbassignments, updates=bscores, shape=tf.shape(scores))

    scores *= score_mask
    scores -= (1. - score_mask) * 1e6
    return scores, score_mask
