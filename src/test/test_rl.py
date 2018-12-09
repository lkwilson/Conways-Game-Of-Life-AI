import os
import unittest

from cgolai.cgol import Model, CgolProblem
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
        self.filename = os.path.join('src', 'test', 'test_rl_model.dat')

    def test_cgolai_compat(self):
        verbose = False
        if verbose:
            print('start')
        model = Model(size=(3, 3))
        problem = CgolProblem(model)
        rl = RL(problem, [None, 100, 100, 100, None],
                batches=5,
                batch_size=3,
                epsilon_decay_factor=0.9999,
                epsilon_init=0.5,
                max_steps=100,
                replay_count=3)
        rl.train(iterations=100)
        if verbose:
            problem.reset()
            print(problem.model.board)
            import numpy as np
            print(np.array([rl.get_value(action) for action in problem.actions()[:-1]]).reshape(problem.model.size))
            print('end')

    def test_rl_basic(self):
        # build and train
        full_test = False
        if not full_test:
            self.batch_size = 5
            self.batches = 5
            self.replay_count = 5

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
        if full_test:
            self.assertEqual(steps, 7)

    def test_rl_save_load(self):
        # build and train
        full_test = False
        if not full_test:
            self.batch_size = 5
            self.batches = 5
            self.replay_count = 5

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
                optim=None,
                filename=self.filename)
        rl.train(batches=self.batches,
                 batch_size=self.batch_size,
                 replay_count=self.replay_count,
                 iterations=self.iterations,
                 epsilon=self.epsilon_init)
        rl.save()
        rl.load()

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
        if full_test:
            self.assertEqual(steps, 7)

    def test_rl_save_load_new_rl(self):
        # build and train
        full_test = False
        if not full_test:
            self.batch_size = 5
            self.batches = 5
            self.replay_count = 5

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
                optim=None,
                filename=self.filename)
        rl.train(batches=self.batches,
                 batch_size=self.batch_size,
                 replay_count=self.replay_count,
                 iterations=self.iterations,
                 epsilon=self.epsilon_init)
        rl.save()
        rl = RL(self.problem, filename=self.filename, load=True)

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
        if full_test:
            self.assertEqual(steps, 7)

