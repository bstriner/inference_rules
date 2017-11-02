import numpy as np
from keras.optimizers import Adam

from inference_rules.inference_model import InferenceModel
from inference_rules.initializers import uniform_initializer
from inference_rules.parser import load_pickle


def main():
    dataset_path = 'output/dataset-WN18'
    output_path = 'output/inference_model-WN18'
    data = np.load('{}/dataset.npz'.format(dataset_path))
    entities = load_pickle('{}/entities.pickle'.format(dataset_path))
    relations = load_pickle('{}/relations.pickle'.format(dataset_path))
    train = data['train']
    valid = data['valid']

    # Hyperparameters
    opt = Adam(3e-4)
    rule_n = 100
    rule_depth = 3
    splits = 10
    tau0 = 1
    tau_decay = 1e-6
    tau_min = 0.25
    epochs = 100
    batches = 1024
    val_batches = 100
    pos_samples = 64
    neg_target_samples = 64
    neg_source_samples = 64
    initializer = uniform_initializer(0.05)

    model = InferenceModel(train=train,
                           valid=valid,
                           rule_n=rule_n,
                           rule_depth=rule_depth,
                           entities=entities,
                           relations=relations,
                           opt=opt,
                           initializer=initializer,
                           tau0=tau0,
                           tau_min=tau_min,
                           tau_decay=tau_decay,
                           splits=splits)
    model.train(
        output_path=output_path,
        epochs=epochs,
        batches=batches,
        val_batches=val_batches,
        pos_samples=pos_samples,
        neg_target_samples=neg_target_samples,
        neg_source_samples=neg_source_samples)


if __name__ == '__main__':
    main()
