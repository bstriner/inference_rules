import os

import numpy as np

from inference_rules.parser import read_tuples


def covariance_calc(path):
    pass


def rel_name(rels, i):
    if i < len(rels):
        return rels[i]
    else:
        return rels[i - len(rels)] + "^-1"


def filter_file(path, outpath, selected):
    with open(outpath, 'w') as f:
        for tup in read_tuples(path):
            if tup[1] in selected:
                f.write("{}\t{}\t{}\n".format(tup[0], tup[1], tup[2]))


def main():
    basepath = r'input/ms'
    outpath = 'output/filtered'
    os.makedirs(outpath, exist_ok=True)
    basefiles = ['train', 'test', 'valid']
    files = [os.path.join(basepath, '{}.txt'.format(f)) for f in basefiles]

    # Collect entities
    objects = [set() for _ in range(3)]
    for f in files:
        for tup in read_tuples(f):
            for i in range(3):
                objects[i].add(tup[i])
    objects = [list(o) for o in objects]
    for o in objects:
        o.sort()
    subjects = objects[0]
    rels = objects[1]
    targets = objects[2]
    entities = list(set(targets + subjects))
    entities.sort()

    # Relationship list
    reln = len(rels)
    rmap = {r: i for i, r in enumerate(rels)}
    emap = {r: i for i, r in enumerate(entities)}

    # graph map
    mapf = {}
    for f in files:
        for tup in read_tuples(f):
            s = emap[tup[0]]
            r = rmap[tup[1]]
            t = emap[tup[2]]
            if s not in mapf:
                mapf[s] = {}
            if t not in mapf[s]:
                mapf[s][t] = []
            mapf[s][t].append(r)
            if t not in mapf:
                mapf[t] = {}
            if s not in mapf[t]:
                mapf[t][s] = []
            mapf[t][s].append(r + reln)

    # Covariance map
    mat = np.zeros((reln * 2, reln * 2))
    for e1 in mapf.keys():
        for e2 in mapf[e1].keys():
            rs = mapf[e1][e2]
            for r1 in rs:
                for r2 in rs:
                    mat[r1, r2] += 1

    np.save('output/cov.npy', mat)
    thresh = 0.95
    p = mat / np.expand_dims(np.diag(mat), 1)
    selected = set(range(reln))
    dupes = 0
    with open('output/dupes.txt', 'w') as f:
        for r1 in range(reln):
            for r2 in range(r1 + 1, reln * 2):
                if p[r1, r2] > thresh and p[r2, r1] > thresh:
                    dupes += 1
                    f.write("{}=={}\n".format(rel_name(rels, r1), rel_name(rels, r2)))
                    if r2 < reln:
                        r3 = r2
                    else:
                        r3 = r2 - reln
                    if r1 in selected and r3 in selected:
                        selected.remove(r1)

    print("Selected")
    selected = list(selected)
    selected.sort()
    with open('output/selected.txt', 'w') as f:
        for s in selected:
            f.write("{}\n".format(rels[s]))
    print("Selection {}->{}".format(reln, len(selected)))
    print("Dupes {}".format(dupes))
    sel = [rels[s] for s in selected]
    for f, b in zip(files, basefiles):
        filter_file(f, os.path.join(outpath, "{}.txt".format( b)), selected=sel)


if __name__ == '__main__':
    main()
