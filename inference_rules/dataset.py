import os

import numpy as np
import tensorflow as tf
from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import Features, Feature
from tensorflow.python.lib.io.tf_record import TFRecordWriter
from tqdm import tqdm

from .parser import load_pickle
from .preprocessing import FILE_NAMES


def load_dataset(input_path):
    data = [np.load(os.path.join(input_path, '{}.npy'.format(f)))
            for f in FILE_NAMES]
    return data


def load_relationships(input_path):
    return load_pickle(os.path.join(input_path, 'relations.pickle'))


def load_entities(input_path):
    return load_pickle(os.path.join(input_path, 'entities.pickle'))


def candidate_map(tups, r_k):
    ret = [set() for _ in range(r_k)]
    for s, r, t in tups:
        ret[r].add(t)
    return ret


def add_to_map(data, s, r, t):
    if t not in data[s]:
        data[s][t] = set()
    data[s][t].add(r)


def relation_map(tups, e_k, r_k):
    forward = {i: {} for i in range(e_k)}
    backward = {i: {} for i in range(e_k)}
    for s, r, t in tups:
        add_to_map(forward, s, r, t)
        add_to_map(forward, t, r + r_k, s)
        add_to_map(backward, t, r, s)
        add_to_map(backward, s, r + r_k, t)
    return forward, backward


def calc_neighbors(data):
    n = {k: set(v.keys()) for k, v in data.items()}
    return n


def feature_float32(value):
    return Feature(float_list=tf.train.FloatList(value=value.flatten()))


def feature_int64(value):
    return Feature(int64_list=tf.train.Int64List(value=value.flatten()))


def feature_bytes(value):
    return Feature(bytes_list=tf.train.BytesList(value=[value]))


def feature_array(arr):
    arr = arr.astype(np.int32)
    shp = np.array(arr.shape, np.int32)
    return feature_bytes(arr.flatten().tobytes()), feature_int64(shp)


def calc_correct_map(tups):
    ret = {}
    for s, r, t in tups:
        if s not in ret:
            ret[s] = {}
        if r not in ret[s]:
            ret[s][r] = set()
        ret[s][r].add(t)
    return ret


def collate_walks(walks):
    n = len(walks)
    m = max(w.shape[0] for w in walks)
    o = walks[0].shape[1]
    ret = np.zeros((n, m, o), dtype=np.int32)
    for i, w in enumerate(walks):
        assert w.ndim == 2
        ret[i, :w.shape[0], :w.shape[1]] = w
    return ret


def select_subset(es, n):
    m = len(es)
    if m > n:
        idx = np.random.choice(np.arange(m), size=(n,), replace=False)
        return [es[i] for i in idx]
    else:
        return es


def assignment_array(n, i, pos):
    return np.array([[i, pos]], dtype=np.int32).repeat(n, axis=0)


class DatasetProcessor(object):
    def __init__(self, input_path, params, negative_samples=200, mode='train'):
        self.mode = mode
        self.params = params
        self.data = load_dataset(input_path)
        self.train, self.valid, self.test = self.data
        if mode == 'train':
            self.current = self.train
            self.all_data = self.train
        elif mode == 'valid':
            self.current = self.valid
            self.all_data = np.concatenate((self.train, self.valid), axis=0)
        elif mode == 'test':
            self.current = self.test
            self.all_data = np.concatenate((self.train, self.valid, self.test), axis=0)
        else:
            raise ValueError()
        self.relationships = load_relationships(input_path)
        self.entities = load_entities(input_path)
        self.r_k = len(self.relationships)
        self.e_k = len(self.entities)
        self.candidates = candidate_map(self.all_data, self.r_k)
        self.forward, self.backward = relation_map(self.train, self.e_k, self.r_k)
        self.forward_neighbors = calc_neighbors(self.forward)
        self.backward_neighbors = calc_neighbors(self.backward)
        self.correct_map = calc_correct_map(self.all_data)
        self.negative_samples = negative_samples

    def generate_dataset(self, output_path):
        if os.path.exists(output_path):
            raise ValueError("Dataset already exists: {}".format(output_path))
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        trivial = 0
        records = 0
        fact_id = 0
        with TFRecordWriter(output_path) as writer:
            for s, r, t in tqdm(self.current):
                for features in self.generate_features(s, r, t, fact_id=fact_id):
                    if features is None:
                        trivial += 1
                    else:
                        example = Example(features=Features(feature=features))
                        writer.write(example.SerializeToString())
                        records += 1
                fact_id += 1
        with open(output_path + ".txt", 'w') as f:
            f.write("Trivial: {}\n".format(trivial))
            f.write("Records: {}\n".format(records))

    def get_correct(self, s, r):
        if s in self.correct_map:
            if r in self.correct_map[s]:
                return self.correct_map[s][r]
        return set()

    def get_candidates(self, r):
        return self.candidates[r]

    def generate_walks_1(self, s, r, t):
        walks = []
        if t in self.forward[s]:
            for rel in self.forward[s][t]:
                if rel != r:
                    walks.append([rel])
        if len(walks) > 0:
            return np.array(walks, dtype=np.int32)
        else:
            return np.zeros((1, 1), dtype=np.int32)

    def generate_walks_2(self, s, r, t):
        nf = self.forward_neighbors[s]
        nb = self.backward_neighbors[t]
        ne = list(nf.intersection(nb))
        n = len(ne)
        walks = []
        for i in range(n):
            z = ne[i]
            rels1 = self.forward[s][z]
            rels2 = self.backward[t][z]
            for rel1 in rels1:
                for rel2 in rels2:
                    if not ((rel1 == r and z == t) or (rel2 == r and z == s)):
                        walks.append([rel1, rel2])
        if len(walks) > 0:
            return np.array(walks, dtype=np.int32)
        else:
            return np.zeros((1, 2), dtype=np.int32)

    def generate_feats_forward(self, s, r, t):
        feats = []
        for x, y in self.forward[s].items():
            for z in y:
                if not (x == t and z == r):
                    feats.append([x, z])
        if len(feats) > 0:
            return np.array(feats, np.int32)
        else:
            return np.zeros((1, 2), np.int32)

    def generate_feats_backward(self, s, r, t):
        feats = []
        for x, y in self.backward[t].items():
            for z in y:
                if not (x == s and z == r):
                    feats.append([x, z])
        if len(feats) > 0:
            return np.array(feats, np.int32)
        else:
            return np.zeros((1, 2), np.int32)

    def generate_tup_features(self, s, r, t):
        w1 = self.generate_walks_1(s, r, t)
        w2 = self.generate_walks_2(s, r, t)
        return w1, w2

    def generate_examples(self, batch_size):
        ss = []
        rs = []
        ts = []
        tnegs = []
        tnegassignments = []
        w1s = []
        w1assignments = []
        w2s = []
        w2assignments = []
        ffs = []
        ffassignments = []
        fbs = []
        fbassignments = []
        index = 0
        while index < batch_size:
            id = np.random.randint(low=0, high=self.current.shape[0] - 1)
            s, r, t = self.current[id, :]
            type_match = self.get_candidates(r)
            correct = self.get_correct(s, r)
            candidates = list(type_match - correct)
            n = len(candidates)
            if n > 0:
                ss.append(s)
                rs.append(r)
                ts.append(t)
                if self.params.enable_walks1:
                    w1 = self.generate_walks_1(s, r, t)
                    w1s.append(w1)
                    w1assignments.append(assignment_array(n=w1.shape[0], i=index, pos=0))
                if self.params.enable_walks2:
                    w2 = self.generate_walks_2(s, r, t)
                    w2s.append(w2)
                    w2assignments.append(assignment_array(n=w2.shape[0], i=index, pos=0))
                if self.params.enable_secondary:
                    ff = self.generate_feats_forward(s, r, t)
                    fb = self.generate_feats_backward(s, r, t)
                    ffs.append(ff)
                    ffassignments.append(assignment_array(n=ff.shape[0], i=index, pos=0))
                    fbs.append(fb)
                    fbassignments.append(assignment_array(n=fb.shape[0], i=index, pos=0))

                ns = select_subset(candidates, self.negative_samples)
                fact_pos = 1
                for c in ns:
                    tnegs.append(c)
                    tnegassignments.append(assignment_array(n=1, i=index, pos=fact_pos))
                    if self.params.enable_walks1:
                        w1 = self.generate_walks_1(s, r, c)
                        w1s.append(w1)
                        w1assignments.append(assignment_array(n=w1.shape[0], i=index, pos=fact_pos))
                    if self.params.enable_walks2:
                        w2 = self.generate_walks_2(s, r, c)
                        w2s.append(w2)
                        w2assignments.append(assignment_array(n=w2.shape[0], i=index, pos=fact_pos))
                    if self.params.enable_secondary:
                        ff = self.generate_feats_forward(s, r, c)
                        fb = self.generate_feats_backward(s, r, c)
                        ffs.append(ff)
                        ffassignments.append(assignment_array(n=ff.shape[0], i=index, pos=fact_pos))
                        fbs.append(fb)
                        fbassignments.append(assignment_array(n=fb.shape[0], i=index, pos=fact_pos))
                    fact_pos += 1
                index += 1

        ss = np.array(ss, dtype=np.int32)
        rs = np.array(rs, dtype=np.int32)
        ts = np.array(ts, dtype=np.int32)
        tnegs = np.array(tnegs, dtype=np.int32)
        tnegassignments = np.concatenate(tnegassignments, axis=0)
        kw = {
            's': ss,
            'r': rs,
            't': ts,
            'tneg': tnegs,
            'tnegassignments': tnegassignments
        }
        if self.params.enable_walks1:
            w1s = np.concatenate(w1s, axis=0)
            w1assignments = np.concatenate(w1assignments, axis=0)
            kw['w1s'] = w1s
            kw['w1assignments'] = w1assignments
        if self.params.enable_walks2:
            w2s = np.concatenate(w2s, axis=0)
            w2assignments = np.concatenate(w2assignments, axis=0)
            kw['w2s'] = w2s
            kw['w2assignments'] = w2assignments
        if self.params.enable_secondary:
            ffs = np.concatenate(ffs, axis=0)
            ffassignments = np.concatenate(ffassignments, axis=0)
            fbs = np.concatenate(fbs, axis=0)
            fbassignments = np.concatenate(fbassignments, axis=0)
            kw['ffs'] = ffs
            kw['ffassignments'] = ffassignments
            kw['fbs'] = fbs
            kw['fbassignments'] = fbassignments
        return kw

    def generate_features(self, s, r, t, fact_id):
        ts = []
        w1s = []
        w2s = []
        ffs = []
        fbs = []

        w1, w2 = self.generate_tup_features(s, r, t)
        ff = self.generate_feats_forward(s, r, t)
        fb = self.generate_feats_backward(s, r, t)

        type_match = self.get_candidates(r)
        correct = self.get_correct(s, r)
        candidates = list(type_match - correct)

        n = len(candidates)
        if n == 0:
            yield None
        else:
            if self.mode == 'test':
                feats = {
                    's': feature_int64(s),
                    'r': feature_int64(r),
                    't': feature_int64(t),
                    'fact_id': feature_int64(np.array(fact_id, dtype=np.int32)),
                    'fact_pos': feature_int64(np.array(0, dtype=np.int32))
                }
                feats['w1'], feats['w1shape'] = feature_array(w1)
                feats['w2'], feats['w2shape'] = feature_array(w2)
                feats['ff'], feats['ffshape'] = feature_array(ff)
                feats['fb'], feats['fbshape'] = feature_array(fb)
                yield feats
            else:
                ts.append(t + 1)
                w1s.append(w1)
                w2s.append(w2)
                ffs.append(ff)
                fbs.append(fb)
            assert n > 0
            ns = select_subset(candidates, self.negative_samples)
            fact_pos = 1
            for c in ns:
                w1, w2 = self.generate_tup_features(s, r, c)
                ff = self.generate_feats_forward(s, r, c)
                fb = self.generate_feats_backward(s, r, c)
                if self.mode == 'test':
                    feats = {
                        's': feature_int64(s),
                        'r': feature_int64(r),
                        't': feature_int64(c),
                        'fact_id': feature_int64(np.array(fact_id, dtype=np.int32)),
                        'fact_pos': feature_int64(np.array(fact_pos, dtype=np.int32))
                    }
                    feats['w1'], feats['w1shape'] = feature_array(w1)
                    feats['w2'], feats['w2shape'] = feature_array(w2)
                    feats['ff'], feats['ffshape'] = feature_array(ff)
                    feats['fb'], feats['fbshape'] = feature_array(fb)
                    yield feats
                    fact_pos += 1
                else:
                    ts.append(c + 1)
                    w1s.append(w1)
                    w2s.append(w2)
                    ffs.append(ff)
                    fbs.append(fb)

            if self.mode != 'test':
                ts = np.array(ts, dtype=np.int32)
                w1s = collate_walks(w1s)
                w2s = collate_walks(w2s)
                ffs = collate_walks(ffs)
                fbs = collate_walks(fbs)

                feat = {}
                feat['s'] = feature_int64(s)
                feat['r'] = feature_int64(r)
                feat['t'], _ = feature_array(ts)
                feat['w1'], feat['w1shape'] = feature_array(w1s)
                feat['w2'], feat['w2shape'] = feature_array(w2s)
                feat['ff'], feat['ffshape'] = feature_array(ffs)
                feat['fb'], feat['fbshape'] = feature_array(fbs)

                yield feat
