import tensorflow as tf


def embedding_var(x_k, name, params):
    initializer = tf.initializers.random_uniform(-0.05, 0.05)
    with tf.variable_scope(name):
        embedding = tf.get_variable(
            name=name,
            shape=[x_k, params.units],
            trainable=True,
            dtype=tf.float32,
            initializer=initializer)
        return embedding


class GraphEmbedding(object):
    def __init__(self, e_k, r_k, params):
        self.e_embed = embedding_var(e_k, 'entity_embedding', params=params)
        self.r_embed = embedding_var(r_k, 'relation_embedding', params=params)

    def embed_entity(self, e):
        return tf.nn.embedding_lookup(self.e_embed, e)

    def embed_relation(self, r):
        return tf.nn.embedding_lookup(self.r_embed, r)
