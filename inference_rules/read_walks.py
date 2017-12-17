import pickle

import numpy as np


def read_walks(path, n):
    walks = []
    for i in range(n):
        with open(path.format(i), 'rb') as f:
            walks.extend(pickle.load(f))
    return walks


def yield_walks(path, n):
    for i in range(n):
        with open(path.format(i), 'rb') as f:
            for w in pickle.load(f):
                yield w


def count_walks(gen):
    depth = 2
    pos_walks = [0 for _ in range(depth)]
    neg_walks = [0 for _ in range(depth)]
    for w in gen:
        for i, d in enumerate(w['rels']):
            if d is not None:
                pos_walks[i] += d.shape[0]
        for n in w['neg_walks']:
            for i, d in enumerate(n['rels']):
                if d is not None:
                    neg_walks[i] += d.shape[0]
    print("Pos walks: {}".format(pos_walks))
    print("Neg walks: {}".format(neg_walks))
    return pos_walks, neg_walks


def merge_walks(path, n, depth):
    pos_walk_counts, neg_walk_counts = count_walks(yield_walks(path, n))
    pos_walks = [np.zeros((pos_walk_counts[i], i + 1), dtype=np.int32) for i in range(depth)]
    neg_walks = [np.zeros((neg_walk_counts[i], i + 1), dtype=np.int32) for i in range(depth)]
    pos_idx = [0 for i in range(depth)]
    neg_idx = [0 for i in range(depth)]
    walk_idx = []
    for w in yield_walks(path, n):
        ids = []
        for i, d in enumerate(w['rels']):
            if d is None:
                ids.append(None)
            else:
                id = slice(pos_idx[i], pos_idx[i] + d.shape[0])
                ids.append(id)
                pos_walks[i][id, :] = d
                pos_idx[i] += d.shape[0]
        neg_ids = []
        for n in w['neg_walks']:
            nids = []
            for i, d in enumerate(n['rels']):
                if d is None:
                    nids.append(None)
                else:
                    id = slice(neg_idx[i], neg_idx[i] + d.shape[0])
                    nids.append(id)
                    neg_walks[i][id, :] = d
                    neg_idx[i] += d.shape[0]
            neg_ids.append(nids)
        walk_idx.append({"ids": ids, "neg_ids": neg_ids, 'r': w['r']})

    return pos_walks, neg_walks, walk_idx
