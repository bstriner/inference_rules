import tensorflow as tf

from .util import leaky_relu


def feature_secondary_masked(
        secondary_ent_embedded,
        rel_embedded,
        target_embedded,
        query_embedded,
        mask,
        params,
        reuse=tf.AUTO_REUSE):
    """

    :param secondary_ent_embedded: (n*samples, walks, units)
    :param rel_embedded: (n*samples, walks, units)
    :param query_embedded: (n*samples, units)
    :param target_embedded: (n*samples, units)
    :param mask:
    :param params:
    :param reuse:
    :return:
    """
    walk_count = tf.shape(secondary_ent_embedded)[1]
    q_tiled = tf.tile(tf.expand_dims(query_embedded, 1), [1, walk_count, 1])
    q_flat = tf.reshape(q_tiled, (-1, params.units))
    e_flat = tf.reshape(secondary_ent_embedded, (-1, params.units))
    r_flat = tf.reshape(rel_embedded, (-1, params.units))
    t_tiled = tf.tile(tf.expand_dims(target_embedded, 1), [1, walk_count, 1])
    t_flat = tf.reshape(t_tiled, (-1, params.units))

    scores_flat = feature_secondary(
        ent_embedded=e_flat,
        rel_embedded=r_flat,
        query_embedded=q_flat,
        target_embedded=t_flat,
        params=params,
        reuse=reuse
    )  # (n*samples*walks,)
    scores = tf.reshape(scores_flat, (-1, walk_count))
    mask = tf.reduce_sum(mask, 2)
    print("Scores: {}, mask {}".format(scores, mask))
    scores *= mask
    scores = tf.reduce_sum(scores, axis=1)  # (n*samples,)
    return scores


def feature_secondary(ent_embedded, rel_embedded, query_embedded, target_embedded, params, reuse=tf.AUTO_REUSE):
    ectx = tf.layers.dense(
        inputs=ent_embedded,
        units=params.units,
        name='secondary_ent_ctx',
        reuse=reuse
    )
    rctx = tf.layers.dense(
        inputs=rel_embedded,
        units=params.units,
        name='secondary_rel_ctx',
        reuse=reuse
    )
    qctx = tf.layers.dense(
        inputs=query_embedded,
        units=params.units,
        name='secondary_q_ctx',
        reuse=reuse
    )
    tctx = tf.layers.dense(
        inputs=target_embedded,
        units=params.units,
        name='secondary_t_ctx',
        reuse=reuse
    )
    h = ectx + rctx + qctx + tctx
    for i in range(3):
        h = tf.layers.dense(
            inputs=h,
            units=params.units,
            name='secondary_{}'.format(i),
            reuse=reuse
        )
        h = leaky_relu(h)
    scores = tf.layers.dense(
        inputs=h,
        units=1,
        name='secondary_scores',
        reuse=reuse
    )
    return tf.squeeze(scores, 1)  # (n,)
