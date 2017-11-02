import numpy as np

from inference_rules.parser import load_pickle
from inference_rules.walker import augment, count_reachable, triple_map
from tqdm import tqdm

def main():
    dataset_path = 'output/dataset-WN18'
    output_path = 'output/inference_model-WN18'
    data = np.load('{}/dataset.npz'.format(dataset_path))
    entities = load_pickle('{}/entities.pickle'.format(dataset_path))
    relations = load_pickle('{}/relations.pickle'.format(dataset_path))
    train = data['train']
    valid = data['valid']

    atriples = list(augment(train, len(relations)))
    tm = triple_map(atriples)

    r2s = []
    r3s = []
    for i in tqdm(range(len(entities))):
        r2 = count_reachable(tm, 2, i)
        r3 = count_reachable(tm, 3, i)
        print "Entity {} reachable@2 {} reachable@3 {}".format(i, r2, r3)
        r2s.append(r2)
        r3s.append(r3)
    print "Mean@2 {} Mean@3 {}".format(np.mean(r2s), np.mean(r3s))
    print "Max@2 {} Max@3 {}".format(np.max(r2s), np.max(r3s))


if __name__ == '__main__':
    main()
