import unittest
import torch

from cgolai.ai import NNTorch


class TestNN(unittest.TestCase):
    def setUp(self):
        self.inner = [10, 10]
        self.rows = 50
        self.verbose = True  # for debugging
        self.verbose_model = False
        self.iterations = 2

    @staticmethod
    def norm(A):
        total = 0
        for a in A:
            for b in a:
                total += b*b
        return total/len(A)

    def tet_linear(self):
        nn = NN(shape=[1, 1])
        X = [[x] for x in range(16)]
        T = [[row[0]*2] for row in X]
        nn.fit(X, T, verbose=self.verbose_model, iterations=100)
        Y = nn.predict(X)

        # reporting
        T = torch.Tensor(T)
        if self.verbose:
            print('torch nn linear', self.norm(T-Y))
        self.assertTrue(self.norm(T-Y) < 10)  # for debugging

    def tst_fit_multi(self):
        nn = NN([16, *self.inner, 4])
        X = torch.randn(self.rows, 16)
        T = torch.Tensor([[sum(row[i:i+4])/4 for i in range(4)] for row in X])

        X, T = nn.fit(X, T, verbose=self.verbose_model, iterations=self.iterations)
        Y = nn.predict(X)
        if self.verbose:
            print('torch nn multi', self.norm(T-Y))
        self.assertTrue(self.norm(T-Y) < 10)  # for debugging

    def tet_fit_single(self):
        x = torch.linspace(0, 10, 50).reshape(50, 1)
        t = torch.sin(x).reshape(50, 1)
        nn = NN([None, *self.inner, None])
        x, t = nn.fit(x, t, verbose=self.verbose_model, iterations=self.iterations)
        y = nn.predict(x)
        if self.verbose:
            print('torch nn single', self.norm(t-y))
        self.assertTrue(self.norm(t-y) < 10)  # for debugging

