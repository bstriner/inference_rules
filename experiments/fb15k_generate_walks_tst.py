import numpy as np

from inference_rules.generate_walks import generate_validation_data
from inference_rules.parser import load_pickle


def main():
    dataset_path = 'output/FB15K/dataset'
    output_path = 'output/FB15K/processed/test.pickle'

    data = np.load('{}/dataset.npz'.format(dataset_path))
    relations = load_pickle('{}/relations.pickle'.format(dataset_path))
    train = data['train']
    valid = data['valid']
    test = data['test']
    all_facts = np.concatenate((train, valid, test), axis=0)
    r_k = len(relations)
    max_depth = 2
    max_negative_samples = 20
    generate_validation_data(
        output_path=output_path,
        train=train,
        valid=test,
        all_facts=all_facts,
        r_k=r_k,
        max_depth=max_depth,
        max_negative_samples=max_negative_samples)


if __name__ == '__main__':
    main()
