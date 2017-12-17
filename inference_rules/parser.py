import csv
import pickle

import numpy as np


def read_tuples(file, map=[0, 1, 2]):
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                t = list(line.split("\t"))
                assert len(t) == 3
                yield tuple(t[m] for m in map)


def get_entities_gen(ts):
    es = set()
    for t in ts:
        es.add(t[0])
        es.add(t[2])
    return es


def get_entities(ts):
    es = list(get_entities_gen(ts))
    es.sort()
    return es


def get_relations_gen(ts):
    rs = set(t[1] for t in ts)
    return rs


def get_relations(ts):
    es = list(get_relations_gen(ts))
    es.sort()
    return es


def map_triples(ts, emap, rmap):
    n = len(ts)
    x = np.zeros((n, 3), dtype=np.int32)
    for i, t in enumerate(ts):
        x[i, 0] = emap[t[0]]
        x[i, 1] = rmap[t[1]]
        x[i, 2] = emap[t[2]]
    return x


def save_list(path, data):
    with open(path + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Id', 'Name'])
        for i, d in enumerate(data):
            w.writerow([i, d])
    with open(path + '.pickle', 'wb') as f:
        pickle.dump(data, f)


def entity_types(triples, e_k, r_k, source=True):
    types = np.zeros((r_k, e_k), dtype='float32')
    for i in range(triples.shape[0]):
        types[triples[i, 1], triples[i, 0 if source else 2]] = 1
    return types


def entity_type_logits(triples, e_k, r_k, source=True, eps=1e-9):
    return np.log(eps + entity_types(triples, e_k, r_k, source=source))


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
