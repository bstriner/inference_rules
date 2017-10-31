import numpy as np


def uniform_initializer(scale):
    return lambda size: np.random.uniform(low=-scale, high=scale, size=size)
