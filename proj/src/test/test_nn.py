import unittest
import numpy as np
import torch

from cgolai.ai import NN as NNMine
from cgolai.ai import NNTorch
from cgolai.ai import NNGiven


class TestNN(unittest.TestCase):
    def setUp(self):
        self.inner = [10, 10]
        self.rows = 50
        self.iterations = 30
        # for debugging
        self.verbose = False
        self.verbose_model = False

    def test_mine(self):
        self.linear(NNMine, "mine")
        self.fit_multi(NNMine, "mine")
        self.fit_single(NNMine, "mine")

    def test_torch(self):
        self.linear(NNTorch, "torch")
        self.fit_multi(NNTorch, "torch")
        self.fit_single(NNTorch, "torch")

    def test_given(self):
        self.linear(NNGiven, "given")
        self.fit_multi(NNGiven, "given")
        self.fit_single(NNGiven, "given")

    @staticmethod
    def norm(T, Y):
        if isinstance(Y, torch.Tensor):
            Y = Y.data.numpy()
        T = np.array(T)
        Y = np.array(Y)

        A = T-Y
        total = 0
        for a in A:
            for b in a:
                total += b*b
        return total/len(A)

    def linear(self, NN, the_type):
        nn = NN(shape=[1, 1])
        X = [[float(x)] for x in range(100)]
        T = [[row[0]*2] for row in X]
        X, T = nn.fit(X, T, verbose=self.verbose_model, iterations=100)
        Y = nn.predict(X)
        # reporting
        if self.verbose:
            print('{} linear'.format(the_type), self.norm(T, Y))
        #self.assertTrue(self.norm(T-Y) < 10)  # for debugging

    def fit_multi(self, NN, the_type):
        nn = NN([16, *self.inner, 4])
        X = np.random.rand(self.rows*16).reshape((self.rows, 16)).tolist()
        T = np.array([[sum(row[i:i+4])/4 for i in range(4)] for row in X]).tolist()
        X, T = nn.fit(X, T, verbose=self.verbose_model, iterations=self.iterations)
        Y = nn.predict(X)
        if self.verbose:
            print('{} multi'.format(the_type), self.norm(T, Y))
        #self.assertTrue(self.norm(T-Y) < 10)  # for debugging

    def fit_single(self, NN, the_type):
        X = np.linspace(0, 10, 50).reshape(50, 1).tolist()
        T = np.sin(X).reshape(50, 1).tolist()
        nn = NN([None, *self.inner, None])
        X, T = nn.fit(X, T, verbose=self.verbose_model, iterations=self.iterations)
        Y = nn.predict(X)
        if self.verbose:
            print('{} single'.format(the_type), self.norm(T, Y))
        #self.assertTrue(self.norm(t-y) < 10)  # for debugging

