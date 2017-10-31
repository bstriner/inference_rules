import theano
import theano.tensor as T

from .tensor_util import softmax_nd, tensor_one_hot
import numpy as np

def sample_gumbel_np(shape, eps=1e-9):
    rnd = np.random.uniform(size=shape, low=eps, high=1. - eps).astype(np.float32)
    return -np.log(eps - np.log(eps + rnd))


def gumbel_softmax_sample_np(logits, temperature, srng):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel_np(logits.shape, srng=srng)
    return softmax_nd(y / temperature)


def gumbel_argmax(logits, srng, axis=-1):
    g = sample_gumbel(shape=logits.shape, srng=srng)
    return T.argmax(logits + g, axis=axis)


def sample_one_hot(logits, axis=-1):
    g = sample_gumbel_np(shape=logits.shape)
    h = logits + g
    return tensor_one_hot(np.argmax(h, axis=axis), h.shape[axis])

def sample_argmax_np(logits, axis=-1, keepdims=False):
    g = sample_gumbel_np(shape=logits.shape)
    h = logits + g
    return np.argmax(h, axis=axis, keepdims=keepdims)


def gumbel_softmax(logits, temperature, srng, hard=False, axis=-1):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, srng=srng)
    if hard:
        y_hard = tensor_one_hot(T.argmax(y, axis=axis), y.shape[axis])
        y = theano.gradient.zero_grad(y_hard - y) + y
    return y
