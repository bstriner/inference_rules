import itertools

import numpy as np

from inference_rules.parser import read_tuples, get_entities, get_relations, map_triples, save_list
from inference_rules.util import make_dir


def generate_dataset(input_path, output_path, map=range(3)):
    make_dir(output_path)
    files = 'train', 'valid', 'test'
    data = [list(read_tuples(input_path.format(f), map=map)) for f in files]
    entities = [get_entities(d) for d in data]
    relations = [get_relations(d) for d in data]
    for i, d in enumerate(data):
        print("{}: tuples={}, entities={}, relations={}".format(files[i], len(d), len(entities[i]), len(relations[i])))

    for i in [1, 2]:
        newentities = set(e for e in entities[i] if e not in entities[0])
        newrelations = set(r for r in relations[i] if r not in relations[0])
        print("New in {}: entities={}, relations={}".format(files[i], len(newentities), len(newrelations)))

    all_entities = list(set(itertools.chain.from_iterable(entities)))
    all_entities.sort()

    all_relations = list(set(itertools.chain.from_iterable(relations)))
    all_relations.sort()

    entities_map = {k: i for i, k in enumerate(all_entities)}
    relations_map = {k: i for i, k in enumerate(all_relations)}

    triples = [map_triples(d, entities_map, relations_map) for d in data]

    e_k = len(all_entities)
    r_k = len(all_relations)

    np.random.shuffle(triples[0])

    savedata = {'train': triples[0],
                'valid': triples[1],
                'test': triples[2],
                'e_k': e_k,
                'r_k': r_k}

    np.savez('{}/dataset.npz'.format(output_path), **savedata)
    save_list('{}/entities'.format(output_path), all_entities)
    save_list('{}/relations'.format(output_path), all_relations)
