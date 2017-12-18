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
    def __init__(self, e_k, r_k, params, training):
        self.e_embed = embedding_var(e_k, 'entity_embedding', params=params)
        self.r_embed = embedding_var(r_k, 'relation_embedding', params=params)
        self.input_dropout = params.input_dropout
        self.training = training

    def embed_entity(self, e):
        h = tf.nn.embedding_lookup(self.e_embed, e)
        if self.input_dropout > 0:
            h = tf.layers.dropout(h, rate=self.input_dropout, training=self.training)
        return h

    def embed_relation(self, r):
        h = tf.nn.embedding_lookup(self.r_embed, r)
        if self.input_dropout > 0:
            h = tf.layers.dropout(h, rate=self.input_dropout, training=self.training)
        return h
