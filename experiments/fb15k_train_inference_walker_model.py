import pickle

import numpy as np
from keras.optimizers import Adam

from inference_rules.inference_walker_model import InferenceWalkerModel
from inference_rules.initializers import uniform_initializer
from inference_rules.parser import load_pickle
from keras.regularizers import l1_l2
from inference_rules.read_walks import read_walks

def main():
    dataset_path = 'output/FB15K/dataset'
    output_path = 'output/FB15K/inference_walker'
    data = np.load('{}/dataset.npz'.format(dataset_path))
    entities = load_pickle('{}/entities.pickle'.format(dataset_path))
    relations = load_pickle('{}/relations.pickle'.format(dataset_path))
    train = data['train']
    valid = data['valid']
    test = data['test']
    train_path = 'output/FB15K/train_walks/walks-{}.npz'
    train_splits = 20
    train_walks = read_walks(train_path, train_splits)

    regularizer = l1_l2(1e-5, 1e-7)

    # Hyperparameters
    opt = Adam(3e-4)
    tau0 = 1
    tau_decay = 1e-6
    tau_min = 0.25
    epochs = 100
    batches = 1024
    val_batches = 100
    initializer = uniform_initializer(0.05)
    rule_counts = [256, 256]

    model = InferenceWalkerModel(
        relations=relations,
        opt=opt,
        rule_counts=rule_counts,
        initializer=initializer,
        tau0=tau0,
        tau_min=tau_min,
        regularizer=regularizer,
        tau_decay=tau_decay)
    model.train(
        output_path=output_path,
        epochs=epochs,
        batches=batches,
        val_batches=val_batches,
        train_walks=train_walks)


if __name__ == '__main__':
    main()
