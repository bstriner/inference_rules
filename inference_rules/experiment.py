from tensorflow.contrib.learn import Experiment
from tensorflow.python.estimator.estimator import Estimator

from .dataset import load_entities, load_relationships
from .feed_input import make_input_fn, FeedFnHook
from .model import make_model_fn
import tensorflow as tf

def experiment_fn(run_config, hparams):
    input_path = 'output/dataset'
    #train_path = 'output/processed/train.tfrecords'
    #val_path = 'output/processed/valid.tfrecords'
    relations = load_relationships('output/dataset')
    entities = load_entities('output/dataset')
    train_hook = FeedFnHook(
        input_path=input_path,
        mode='train',
        batch_size=tf.flags.FLAGS.batch_size,
        params=hparams,
        negative_samples=hparams.negative_samples)
    eval_hook = FeedFnHook(
        input_path=input_path,
        mode='valid',
        batch_size=tf.flags.FLAGS.batch_size,
        params=hparams,
        negative_samples=hparams.negative_samples)
    estimator = Estimator(
        model_fn=make_model_fn(entities=entities, relations=relations),
        config=run_config,
        params=hparams)
    experiment = Experiment(
        estimator=estimator,
        train_input_fn=make_input_fn(hparams),
        eval_input_fn=make_input_fn(hparams),
        eval_hooks=[eval_hook],
        train_monitors=[train_hook]
    )

    return experiment
