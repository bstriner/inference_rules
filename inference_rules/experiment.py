from tensorflow.contrib.learn import Experiment
from tensorflow.python.estimator.estimator import Estimator

from .dataset import load_entities, load_relationships
from .input import make_input_fn
from .model import make_model_fn


def experiment_fn(run_config, hparams):
    train_path = 'output/processed/train.tfrecords'
    val_path = 'output/processed/valid.tfrecords'
    relations = load_relationships('output/dataset')
    entities = load_entities('output/dataset')
    estimator = Estimator(
        model_fn=make_model_fn(entities=entities, relations=relations),
        config=run_config,
        params=hparams)
    experiment = Experiment(
        estimator=estimator,
        train_input_fn=make_input_fn(train_path),
        eval_input_fn=make_input_fn(val_path)
    )

    return experiment
