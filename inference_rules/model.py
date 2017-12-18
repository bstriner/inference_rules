import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer, apply_regularization

from .inference import inference_fn, inference_fn_pred
from .losses import cross_entropy_loss
from .util import softmax_nd_masked


def calc_rank(scores):
    pos1 = tf.expand_dims(scores[:, 0], 1)  # (n,1)
    rank = tf.reduce_sum(tf.cast(tf.greater(scores, pos1), tf.float32), axis=1)  # (n,)
    return rank


def make_model_fn(entities, relations):
    e_k = len(entities)
    r_k = len(relations)

    def model_fn(features, labels, mode, params):

        if mode == tf.estimator.ModeKeys.PREDICT:
            scores = inference_fn_pred(
                features=features,
                e_k=e_k,
                r_k=r_k,
                mode=mode,
                params=params)
            predictions = {
                'scores': scores
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        else:
            scores, mask = inference_fn(
                features=features,
                e_k=e_k,
                r_k=r_k,
                mode=mode,
                params=params)
            ranks = calc_rank(scores)
            reciprocal_ranks = 1. / (ranks+1)
            hits_at_1 = tf.cast(tf.equal(ranks, 0), tf.float32)
            hits_at_10 = tf.cast(tf.less(ranks, 10), tf.float32)

            # Loss
            probs = softmax_nd_masked(scores, mask=mask, axis=-1)  # (n, samples)
            targets = tf.one_hot(
                tf.zeros((tf.shape(probs)[0],), dtype=tf.int32),
                depth=tf.shape(probs)[1],
                axis=1,
                dtype=tf.float32)
            loss = tf.reduce_mean(cross_entropy_loss(
                labels=targets,
                p=probs,
                params=params,
                mask=mask), 0)

            # Regularization
            if params.l2 > 0:
                reg = apply_regularization(l2_regularizer(params.l2), tf.trainable_variables())
                tf.summary.scalar("regularization", reg)
                loss += reg

            if mode == tf.estimator.ModeKeys.TRAIN:
                tf.summary.scalar('mean_reciprocal_rank', tf.reduce_mean(reciprocal_ranks))
                tf.summary.scalar('mean_rank', tf.reduce_mean(ranks))
                tf.summary.scalar('hits_at_1', tf.reduce_mean(hits_at_1))
                tf.summary.scalar('hits_at_10', tf.reduce_mean(hits_at_10))
                lr = tf.train.exponential_decay(params.lr,
                                                decay_rate=params.decay_rate,
                                                decay_steps=params.decay_steps,
                                                global_step=tf.train.get_global_step(),
                                                name='learning_rate',
                                                staircase=False)
                tf.summary.scalar('learning_rate', lr)
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                print("Trainable: {}".format(list(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))))
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            else:
                eval_metric_ops = {
                    'eval_mean_reciprocal_rank': tf.metrics.mean(reciprocal_ranks),
                    'eval_mean_rank': tf.metrics.mean(ranks),
                    'eval_hits_at_1': tf.metrics.mean(hits_at_1),
                    'eval_hits_at_10': tf.metrics.mean(hits_at_10)
                }
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return model_fn
