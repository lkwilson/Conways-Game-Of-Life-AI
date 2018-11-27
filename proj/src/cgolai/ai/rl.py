import numpy as np

from .nn_torch import NNTorch as NN
#from .nn import NN


class RL:
    def __init__(self, problem, shape, verbose=False, rho=0.1, epsilon_decay_factor=0.9, epsilon_init=1.0,
                 use_nn=True, stochastic=False, **nn_config):
        self._rho = rho
        self._epsilon = epsilon_init
        self._epsilon_decay_factor = epsilon_decay_factor
        self._problem = problem
        self._argbest = np.argmax if self._problem.maximize else np.argmin
        self._verbose = verbose
        self.stochastic = stochastic
        self._use_nn = use_nn
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
            return self._Q.predict(key)
        else:
            return self._Q.get(key, default_q)

    def choose_best_action(self, actions=None, explore=False):
        """ Return best action with epsilon exploration factor """
        if actions is None:
            actions = self._problem.actions()
        epsilon = self._epsilon if explore else 0.0
        if np.random.rand() < epsilon:
            action = actions[np.random.randint(len(actions))]
            return action, self.get_value(action)
        else:
            values = [self.get_value(action) for action in actions]
            best_index = self._argbest(values)
            return actions[best_index], values[best_index]

    def train(self, runs=50, **nn_fit_args):
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
                    if self._use_nn:
                        if isinstance(t, int):
                            t = [[t]]
                        self._Q.fit(old_key, t, **nn_fit_args)
                    else:
                        self._Q[old_key] = t
                old_reward = reward
                old_key = key
                steps += 1
            if self._verbose:
                print('steps', steps)
            # keep? known but unneeded data point. Fitting this could make it
            # unfit other points, and since it's not needed, that could be
            # unessessary
            # self._Q.fit(key, [[0]], **nn_fit_args)
