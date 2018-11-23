import numpy as np

from .nn import NN


class RL:
    def __init__(self, rho, epsilon_decay_factor, problem, maximize_reward=True, **nn_config):
        self.rho = rho
        self.epsilon = 1.0
        self.epsilon_decay_factor = epsilon_decay_factor
        self.problem = problem
        self.argbest = np.argmax if maximize_reward else np.argmin
        self.Q = NN(**nn_config)

    def choose_best_action(self, actions, explore=False):
        """ Return best action with epsilon exploration factor """
        epsilon = 0.0 if explore else self.epsilon
        if np.random.rand() < epsilon:
            return np.random.choice(actions)
        else:
            values = [self.Q.predict(self.problem.key(action)) for action in actions]
            best_index = self.argbest(values)
            return actions[best_index]

    def prep(self):
        self.epsilon *= self.epsilon_decay_factor
        self.problem.reset()

    def run(self, runs, reruns):
        for _ in range(runs):
            self.prep()
            old_reward = None
            old_key = None
            while not self.problem.is_terminal():
                actions = self.problem.actions()
                action = self.choose_best_action(actions, explore=True)
                key, reward = self.problem.do(action)
                if old_key is not None:
                    td_error = old_reward + self.Q.predict(key) - self.Q.predict(old_key)  # old_reward?
                    t = self.Q.predict(old_key) + self.rho * td_error
                    self.Q.fit(old_key, t, iterations=reruns)
                old_reward = reward
                old_key = key
