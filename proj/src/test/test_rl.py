import unittest

from cgolai.ai import RL
from .util import HanoiNN


class TestRL(unittest.TestCase):
    def setUp(self):
        self.inner = [50, 50, 50]
        self.verbose = False
        self.verbose_model = False
        self.problem = HanoiNN()
        self.epsilon_decay_factor = 0.99
        self.epsilon_init = 1.0
        self.epsilon_min = 0.1
        self.stochastic = False
        self.mu = 0.01
        self.discount_factor = 1.0
        self.batches = 80
        self.batch_size = 20
        self.replay_count = 0
        self.iterations = 10

    def test_rl_basic(self):
        # build and train
        rl = RL(self.problem,
                [None, *self.inner, None],
                verbose=self.verbose_model,
                epsilon_decay_factor=self.epsilon_decay_factor,
                epsilon_init=self.epsilon_init,
                epsilon_min=self.epsilon_min,
                stochastic=self.stochastic,
                mu=self.mu,
                discount_factor=self.discount_factor,
                batches=self.batches,
                batch_size=self.batch_size,
                replay_count=self.replay_count,
                h=None,
                optim=None)
        rl.train(batches=self.batches,
                 batch_size=self.batch_size,
                 replay_count=self.replay_count,
                 iterations=self.iterations,
                 epsilon=self.epsilon_init)

        # test it
        steps = 0
        self.problem.reset()
        while not self.problem.is_terminal():
            action, q_val = rl.choose_best_action(explore=False)
            if self.verbose:
                print('action:', action, 'q_val:', q_val)
            self.problem.do(action)
            steps += 1
            if steps > 100:
                break
        if self.verbose and steps != 7:
            print(steps)
        self.assertEqual(steps, 7)

