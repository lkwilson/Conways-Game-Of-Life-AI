import numpy as np


def relu(x):
    """ returns ReLU(x) """
    return np.maximum(0, x)


def relu_p(x):
    """ returns relu'(x) """
    ret = np.ones(shape=x.shape)
    ret[np.where(x < 0)] = 0
    return ret


def identity(x):
    """ an idempotent function """
    return np.copy(x)


def identity_p(x):
    """ its derivative """
    return np.ones(shape=x.shape)


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def sigmoid_p(x):
    return sigmoid(x)*(1 - sigmoid(x))
