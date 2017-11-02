import numpy as np

from inference_rules.parser import load_pickle
from inference_rules.walker import augment, count_reachable, triple_map, entities_ending_on
from tqdm import tqdm

def main():
    dataset_path = 'output/dataset-FB15K'

    data = np.load('{}/dataset.npz'.format(dataset_path))
    entities = load_pickle('{}/entities.pickle'.format(dataset_path))
    relations = load_pickle('{}/relations.pickle'.format(dataset_path))
    train = data['train']
    valid = data['valid']
    e_k = len(entities)
    r_k = len(relations)
    atriples = list(augment(train, r_k))
    tm = triple_map(atriples)

    r2s = []
    r3s = []
    """
    for i in tqdm(range(e_k)):
        r2 = count_on_path(tm, 2, i)
        r3 = count_on_path(tm, 3, i)
        print "Entity {} reachable@2 {} reachable@3 {}".format(i, r2, r3)
        r2s.append(r2)
        r3s.append(r3)
    """
    for i in tqdm(range(valid.shape[0])):
        r1 = entities_ending_on(tm, 1, valid[i,0], valid[i,2])
        r2 = entities_ending_on(tm, 2, valid[i,0], valid[i,2])
        #r3 = entities_ending_on(tm, 3, valid[i,0], valid[i,2])
        rtot1 = len(r1)
        rtot2 = len(r1|r2)
        #rtot3 = len(r1|r2|r3)
        print "Entity {} reachable@1 {} reachable@2 {}".format(i, rtot1, rtot2)
        r2s.append(rtot2)
        #r3s.append(rtot3)
    print "Mean@2 {}".format(np.mean(r2s))
    print "Max@2 {}".format(np.max(r2s))
    print "0@2 {}".format(np.count_nonzero(r2s))
    print "Coverage@2 {}".format(np.count_nonzero(r2s)/valid.shape[0])


if __name__ == '__main__':
    main()
