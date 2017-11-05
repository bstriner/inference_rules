import sys

import numpy as np

from inference_rules.generate_walks import generate_training_walks
from inference_rules.parser import load_pickle


def main(argv):
    assert len(argv) > 1
    split = int(argv[1])
    print "Processing split {}".format(split)
    dataset_path = 'output/FB15K/dataset'
    split_path = 'output/FB15K/splits/split-{}.npz'
    output_path = 'output/FB15K/train_walks/walks-{}.pickle'
    data = np.load(split_path.format(split))
    relations = load_pickle('{}/relations.pickle'.format(dataset_path))
    r_k = len(relations)
    max_depth = 2
    max_negative_samples = 20
    generate_training_walks(
        output_path=output_path.format(split),
        facts=data['facts'],
        holdout=data['holdout'],
        r_k=r_k,
        max_depth=max_depth,
        max_negative_samples=max_negative_samples)


if __name__ == '__main__':
    main(sys.argv)
