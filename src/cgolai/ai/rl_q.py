import numpy as np


class RLQ:
    def __init__(self, problem, verbose=False, rho=0.1, epsilon_decay_factor=0.9,
                 epsilon_init=1.0, stochastic=False, **nn_config):
        self._rho = rho
        self._epsilon = epsilon_init
        self._epsilon_decay_factor = epsilon_decay_factor
        self._problem = problem
        self._argbest = np.argmax if self._problem.maximize else np.argmin
        self._verbose = verbose
        self.stochastic = stochastic
        self._Q = {}

    def get_value(self, action=None, key=None, default_q=0):
        if key is None:
            key = self._problem.key(action)
        return self._Q.get(key, default_q)

    def choose_best_action(self, actions=None, explore=False):
        """ Return best action with epsilon exploration factor """
        if actions is None:
            actions = self._problem.actions()
        if explore:
            epsilon = self._epsilon
        else:
            epsilon = 0.0

        if np.random.rand() < epsilon:
            action = actions[np.random.randint(len(actions))]
            return action, self.get_value(action)
        else:
            values = [self.get_value(action) for action in actions]
            best_index = self._argbest(values)
            return actions[best_index], values[best_index]

    def train(self, runs=50, max_steps=None):
        for i in range(runs):
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
                    if self._problem.is_terminal():
                        # SARSA requires Q to predict goal state-action pairs as 0
                        if self.stochastic:
                            td_error = old_reward - old_q
                            t = old_q + self._rho * td_error
                        else:
                            t = reward
                    else:
                        td_error = old_reward + q_val - old_q
                        t = old_q + self._rho * td_error
                    self._Q[old_key] = t
                old_reward = reward
                old_key = key
                old_q = q_val
                steps += 1
            if self._verbose:
                print('run', i, 'steps', steps)

