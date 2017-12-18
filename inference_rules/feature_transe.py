import tensorflow as tf


def feature_transe(s, r, t, embedding, params):
    r_ctx = tf.layers.dense(
        inputs=embedding.embed_relation(r),
        units=params.units,
        name='transe_ctx_r',
        reuse=tf.AUTO_REUSE)
    s_ctx = tf.layers.dense(
        inputs=embedding.embed_entity(s),
        units=params.units,
        name='transe_ctx_e',
        reuse=tf.AUTO_REUSE)
    t_ctx = tf.layers.dense(
        inputs=embedding.embed_entity(t),
        units=params.units,
        name='transe_ctx_e',
        reuse=tf.AUTO_REUSE)
    score = -tf.reduce_sum(tf.square(s_ctx + r_ctx - t_ctx), axis=1)
    return score
