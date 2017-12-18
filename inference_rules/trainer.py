import os

import tensorflow as tf
from tensorflow.contrib.learn import RunConfig
from .default_params import get_hparams
from .experiment import experiment_fn

def main(_argv):
    model_dir = tf.flags.FLAGS.model_dir
    os.makedirs(model_dir, exist_ok=True)
    print("model_dir={}".format(model_dir))
    run_config = RunConfig(model_dir=model_dir, save_checkpoints_secs=tf.flags.FLAGS.save_checkpoints_secs)
    hparams = get_hparams(model_dir, create=True)
    estimator = tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=tf.flags.FLAGS.schedule,
        hparams=hparams)
