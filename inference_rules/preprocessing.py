import os

import numpy as np

from .parser import read_tuples, get_entities, get_relations, make_map, map_triples, save_list

FILE_NAMES = ['train', 'valid', 'test']


def write_summary(output_path, entities, relations, data):
    with open(output_path, 'w') as f:
        f.write('Entities: {}\n'.format(len(entities)))
        f.write('Relations: {}\n'.format(len(relations)))
        for n, d in zip(FILE_NAMES, data):
            f.write('{}: {}\n'.format(n, len(d)))


def preprocess(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data = [list(read_tuples(os.path.join(input_dir, f + '.txt'))) for f in FILE_NAMES]

    all_data = []
    for d in data:
        all_data += d

    entities = get_entities(all_data)
    relations = get_relations(all_data)

    write_summary(
        os.path.join(output_dir, 'summary.txt'),
        entities=entities,
        relations=relations,
        data=data)

    emap = make_map(entities)
    rmap = make_map(relations)
    mapped_data = [map_triples(d, emap=emap, rmap=rmap) for d in data]

    for f, d in zip(FILE_NAMES, mapped_data):
        np.save(os.path.join(output_dir, '{}.npy'.format(f)), d)

    save_list(os.path.join(output_dir, 'relations'), relations)
    save_list(os.path.join(output_dir, 'entities'), entities)
