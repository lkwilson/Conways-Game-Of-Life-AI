import numpy as np
import torch

from . import NNTorch as NN


class RL:
    def __init__(self, problem, shape, verbose=False, rho=0.1, epsilon_decay_factor=0.9, epsilon_init=1.0,
                 use_nn=True, stochastic=False, batch_size=10, **nn_config):
        self._rho = rho
        self._epsilon = epsilon_init
        self._epsilon_decay_factor = epsilon_decay_factor
        self._problem = problem
        self._argbest = np.argmax if self._problem.maximize else np.argmin
        self._verbose = verbose
        self.stochastic = stochastic
        self._use_nn = use_nn
        self.batch_size = batch_size
        self.x_batch = []
        self.t_batch = []
        self._trained = False
        self._problem.use_nn = use_nn
        if shape[0] is None or shape[-1] is None:
            shape[0] = self._problem.get_key_dim()
            shape[-1] = 1
        if self._use_nn:
            self._Q = NN(shape, **nn_config)  # mu and shape usually
        else:
            self._Q = {}

    def get_value(self, action=None, key=None, default_q=0):
        if key is None:
            key = self._problem.key(action)
        if self._use_nn:
            return float(self._Q.predict([list(key)])[0])
        else:
            return self._Q.get(key, default_q)

    def choose_best_action(self, actions=None, explore=False):
        """ Return best action with epsilon exploration factor """
        if actions is None:
            actions = self._problem.actions()
        if explore:
            epsilon = self._epsilon
        # elif self._use_nn and not self._trained:
            # epsilon = 1.0
        else:
            epsilon = 0.0

        if np.random.rand() < epsilon:
            action = actions[np.random.randint(len(actions))]
            return action, self.get_value(action)
        else:
            values = [self.get_value(action) for action in actions]
            best_index = self._argbest(values)
            return actions[best_index], values[best_index]

    def train(self, runs=50, max_steps=None, **nn_fit_args):
        """ cleanup this awful method """
        for i in range(runs):
            if self._verbose:
                print('run', i)
            self._epsilon *= self._epsilon_decay_factor
            self._problem.reset()
            old_reward = None
            old_key = None
            steps = 0
            while not self._problem.is_terminal():
                if max_steps is not None and steps > max_steps:
                    break
                action, q_val = self.choose_best_action(explore=True)
                key, reward = self._problem.do(action)
                if steps > 0:
                    old_q = self.get_value(key=old_key)
                    if self._problem.is_terminal():
                        # SARSA requires Q to predict goal state-action pairs as 0,
                        # so I manually implement this due to the inaccuracy of
                        # neural networks
                        if self.stochastic:
                            td_error = old_reward - old_q
                            t = old_q + self._rho * td_error
                        else:
                            t = reward
                    else:
                        q_val = self.get_value(key=key)
                        td_error = old_reward + q_val - old_q
                        t = old_q + self._rho * td_error
                    self.fit(old_key, t, **nn_fit_args)
                old_reward = reward
                old_key = key
                steps += 1
            self.fit(flush=True, iterations=5000)
            if self._verbose:
                print('steps', steps)
            # keep? known but unneeded data point. Fitting this could make it
            # unfit other points, and since it's not needed, that could be
            # unessessary
            # self._Q.fit(key, [[0]], **nn_fit_args)

    def fit(self, x=None, t=None, flush=False, **nn_fit_args):
        if self._use_nn:
            if x is not None and t is not None:
                self.x_batch.append(list(x))
                self.t_batch.append([t])
            if (flush and len(self.x_batch)>0) or len(self.x_batch) >= self.batch_size:
                self._Q.fit(self.x_batch, self.t_batch, **nn_fit_args)
                self.x_batch = []
                self.t_batch = []
                self._trained = True
        else:
            if x is not None and t is not None:
                self._Q[x] = t

