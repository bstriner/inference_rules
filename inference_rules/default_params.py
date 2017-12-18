import json
import os

import numpy as np
import six
import tensorflow as tf
from tensorflow.contrib.training import HParams


def write_hparams_event(model_dir, hparams):
    mdir = model_dir
    with tf.summary.FileWriter(mdir) as f:
        pjson = tf.constant(hparams_string(hparams), tf.string)
        pten = tf.constant(hparams_array(hparams), tf.string)
        with tf.Session() as sess:
            sstr = tf.summary.text('hparams_json', pjson)
            sten = tf.summary.text('hparams_table', pten)
            sumstr, sumten = sess.run([sstr, sten])
        f.add_summary(sumstr)
        f.add_summary(sumten)


def hparams_string(hparams):
    return json.dumps(hparams.values(), indent=4, sort_keys=True)


def hparams_array(hparams):
    d = hparams.values()
    keys = list(d.keys())
    keys.sort()
    values = [str(d[k]) for k in keys]

    arr = np.array([keys, values], dtype=np.string_).transpose((1, 0))
    return arr


def get_hparams(model_dir, create):
    hparams = default_params()
    hparams_path = os.path.join(model_dir, 'configuration-hparams.json')
    if os.path.exists(hparams_path):
        with open(hparams_path) as f:
            hparam_dict = json.load(f)
            for k, v in six.iteritems(hparam_dict):
                setattr(hparams, k, v)
    else:
        if create:
            hparams.parse(tf.flags.FLAGS.hparams)
            with open(hparams_path, 'w') as f:
                json.dump(hparams.values(), f)
            write_hparams_event(model_dir=model_dir, hparams=hparams)
        else:
            raise ValueError("No hparams file found: {}".format(hparams_path))
    return hparams


def default_params():
    return HParams(
        enable_transe=True,
        enable_walks1=False,
        enable_walks2=False,
        enable_secondary=False,
        input_dropout=0.2,
        units=512,
        lr=0.0001,
        decay_rate=0.8,
        decay_steps=50000,
        smoothing=0.1,
        negative_samples=20,
        l2=1e-4)
