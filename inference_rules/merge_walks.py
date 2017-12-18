import os
import pickle

import numpy as np

from inference_rules.read_walks import merge_walks


def write_merged_walks(output_path, walk_path, splits, depth):
    pos_walks, neg_walks, walk_idx = merge_walks(walk_path, splits, depth)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    kw = {}
    for i, w in enumerate(pos_walks):
        kw['pos_walks_{}'.format(i)] = w
        print("pos_walks_{}={}".format(i, w.shape))
    for i, w in enumerate(neg_walks):
        kw['neg_walks_{}'.format(i)] = w
        print("neg_walks_{}={}".format(i, w.shape))
    np.savez(output_path + '.npz', **kw)
    with open(output_path + '.pickle', 'wb') as f:
        pickle.dump(walk_idx, f)
