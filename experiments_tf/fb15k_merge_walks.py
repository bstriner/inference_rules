import os
import pickle

import numpy as np

from inference_rules.parser import load_pickle
from inference_rules.read_walks import merge_walks


def main():
    dataset_path = '../experiments/output/FB15K/dataset'
    output_path = 'output/FB15K/merged_walks'
    data = np.load('{}/dataset.npz'.format(dataset_path))
    entities = load_pickle('{}/entities.pickle'.format(dataset_path))
    relations = load_pickle('{}/relations.pickle'.format(dataset_path))
    train = data['train']
    valid = data['valid']
    test = data['test']
    train_path = '../experiments/output/FB15K/train_walks/walks-{}.pickle'
    train_splits = 20
    depth = 2
    pos_walks, neg_walks, walk_idx = merge_walks(train_path, train_splits, depth)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    kw = {}
    for i, w in enumerate(pos_walks):
        kw['pos_walks_{}'.format(i)] = w
        print("pos_walks_{}={}".format(i, w.shape))
    for i, w in enumerate(neg_walks):
        kw['neg_walks_{}'.format(i)] = w
        print("neg_walks_{}={}".format(i, w.shape))
    np.savez(os.path.join(output_path, 'walks.npz'), **kw)
    with open(os.path.join(output_path, 'walk_idx.pickle'), 'wb') as f:
        pickle.dump(walk_idx, f)


if __name__ == '__main__':
    main()
