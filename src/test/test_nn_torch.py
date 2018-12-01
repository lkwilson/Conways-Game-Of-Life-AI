import unittest
import numpy as np

from cgolai.ai import NNTorch as NN


class TestNN(unittest.TestCase):
    def setUp(self):
        self.inner = [10, 10]
        self.rows = 50
        self.verbose = False  # for debugging
        self.verbose_model = False
        self.iterations = 2

    @staticmethod
    def norm(A):
        total = 0
        for a in A:
            for b in a:
                total += b*b
        return total/len(A)

    def test_passing_batchvs_non(self):
        nn = NN(shape=[5, 5, 2])
        batch = nn.predict(np.array([[1, 2, 3, 4, 5]]))[0]
        single = nn.predict(np.array([1, 2, 3, 4, 5]))
        self.assertEqual(float(batch[0]), float(single[0]))
        self.assertEqual(float(batch[1]), float(single[1]))

    def test_passing_batchvs_non_np(self):
        nn = NN(shape=[5, 5, 2])
        batch = nn.predict([[1, 2, 3, 4, 5]])[0]
        single = nn.predict([1, 2, 3, 4, 5])
        self.assertEqual(float(batch[0]), float(single[0]))
        self.assertEqual(float(batch[1]), float(single[1]))

    def test_general_fit(self):
        x = np.linspace(0, 10, 100).reshape(50, 2)
        t = np.sin(x[:, :1]) + np.sin(x[:, 1:])
        nn = NN([2, 50, 50, 50, 1])
        nn.fit(x, t, verbose=self.verbose_model, iterations=1000)
        y = nn.predict(x)
        if self.verbose:
            print('can nn_torch fit?', np.mean((t-y)**2))
