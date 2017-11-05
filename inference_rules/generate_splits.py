import numpy as np

from .util import make_dir
from .util import split_data


def generate_splits(output_path, train, splits=10):
    make_dir(output_path)
    print type(train)
    train_holdout = list(split_data(train, splits))
    train_facts = list(np.concatenate(list(train_holdout[j] for j in range(splits) if j != i), axis=0)
                       for i in range(splits))
    for i in range(splits):
        holdout = train_holdout[i]
        facts = train_facts[i]
        np.savez('{}/split-{}.npz'.format(output_path, i), holdout=holdout, facts=facts)
