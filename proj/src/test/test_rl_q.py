import unittest

from cgolai.ai import RLQ as RL
from .util import Hanoi


class TestRL(unittest.TestCase):
    def setUp(self):
        self.mu = .001
        self.inner = [10, 10]
        self.rows = 50
        self.verbose = True  # for debugging
        self.iterations = 2
        self.epsilon_decay_factor = 0.9
        self.epsilon_init = 1.0
        self.problem = Hanoi()

    @staticmethod
    def norm(A):
        total = 0
        for a in A:
            for b in a:
                total += b*b
        return total/len(A)

    def test_rl_basic_nonn_nostochastic(self):
        rl = RL(problem=self.problem, stochastic=False, verbose=False, shape=[None, *self.inner, None])
        rl.train(500)
        steps = 0
        self.problem.reset()
        while not self.problem.is_terminal():
            action, _ = rl.choose_best_action(explore=False)
            self.problem.do(action)
            steps += 1
            if steps > 100:
                break
        if self.verbose and steps != 7:
            print(steps)
        self.assertEqual(steps, 7)

    def test_rl_basic_nonn(self):
        rl = RL(problem=self.problem, stochastic=True, verbose=False)
        rl.train(500)
        steps = 0
        self.problem.reset()
        while not self.problem.is_terminal():
            action, _ = rl.choose_best_action(explore=False)
            self.problem.do(action)
            steps += 1
            if steps > 100:
                break
        if self.verbose and steps != 7:
            print(steps)
        self.assertEqual(steps, 7)
