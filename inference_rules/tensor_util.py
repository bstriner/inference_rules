from os.path import exists

import h5py
import keras.backend as K
import theano.tensor as T

from .util import latest_file
from .util import make_path


def softmax_nd(x, axis=-1):
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / T.sum(e_x, axis=axis, keepdims=True)
    return out


def smoothmax_nd(x, axis=-1, keepdims=True):
    sm = softmax_nd(x, axis=axis)
    return T.sum(sm * x, axis=axis, keepdims=keepdims)


def weight_name(i, weight):
    return "param_{}".format(i)


def save_weights(path, weights):
    make_path(path)
    with h5py.File(path, 'w') as f:
        for i, w in enumerate(weights):
            f.create_dataset(name=weight_name(i, w), data=K.get_value(w))


def load_weights(path, weights):
    with h5py.File(path, 'r') as f:
        for i, w in enumerate(weights):
            K.set_value(w, f[weight_name(i, w)])


def load_latest_weights(dir_path, fmt, weights):
    if exists(dir_path):
        path, epoch = latest_file(dir_path, fmt)
        if path:
            print("Loading epoch {}: {}".format(epoch, path))
            load_weights(path, weights)
            return epoch + 1
    return 0


def leaky_relu(x):
    return T.nnet.relu(x, 0.2)


"""
def tensor_one_hot(x, k):
    assert x.ndim == 1
    assert x.dtype == 'int32' or x.dtype == 'int64'
    ret = T.zeros((x.shape[0], k), dtype='float32')
    idx = T.arange(x.shape[0], dtype='int32')
    ret = T.set_subtensor(ret[idx, x], 1.)
    return ret
"""


def tensor_one_hot(x, k, dtype='float32'):
    assert x.dtype == 'int32' or x.dtype == 'int64'
    shape = tuple(x.shape) + (k,)
    ret = T.zeros(shape, dtype=dtype)
    s1 = tuple(slice(dim) for dim in x.shape)
    mgrid = T.mgrid[s1]
    s2 = tuple(mgrid) + (x,)
    ret = T.set_subtensor(ret[s2], 1)
    return ret


def fix_update((a, b), verbose=True):
    if a.dtype != b.dtype:
        print("Mismatch {}: {}->{}".format(a, a.dtype, b.dtype))
        return a, T.cast(b, a.dtype)
    else:
        return a, b


def fix_updates(updates, verbose=True):
    return [fix_update(update, verbose) for update in updates]


def bernoulli_px(p, x):
    # p 0-1
    # x 0 or 1
    assert x.dtype == 'int32' or x.dtype == 'int64'
    return (x * p) + ((1 - x) * (1 - p))


def sample_from_distribution(p, srng):
    assert p.ndim == 2
    cs = T.cumsum(p, axis=1)
    rnd = srng.uniform(low=0., high=1., dtype='float32', size=(p.shape[0],))
    sel = T.sum(T.gt(rnd.dimshuffle((0, 'x')), cs), axis=1)
    sel = T.clip(sel, 0, p.shape[1] - 1)
    return T.cast(sel, 'int32')
