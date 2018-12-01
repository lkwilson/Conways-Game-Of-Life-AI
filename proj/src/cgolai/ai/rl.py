import numpy as np
from . import NNTorch as NN


class RL:
        def __init__(self, problem, shape,
                     verbose=False,
                     epsilon_decay_factor=0.99,
                     epsilon_init=1.0,
                     epsilon_min=0.1,
                     stochastic=False,
                     mu=0.01,
                     discount_factor=1.0,
                     batches=80,
                     batch_size=20,
                     replay_count=0,
                     h=None,
                     optim=None):
            self._epsilon = epsilon_init
            self._epsilon_orig = self._epsilon
            self._epsilon_decay_factor = epsilon_decay_factor
            self._epsilon_min = epsilon_min
            self._problem = problem
            self._argbest = np.argmax if self._problem.maximize else np.argmin
            self._verbose = verbose
            self._stochastic = stochastic
            self._gamma = discount_factor
            self._batches = batches
            self._batch_size = batch_size
            self.batch_size = batch_size
            self.x_batch = []
            self.t_batch = []
            self._replay_count = replay_count
            if shape[0] is None or shape[-1] is None:
                shape[0] = self._problem.get_key_dim()
                shape[-1] = 1
            self._q = NN(shape, mu=mu, h=h, optim=optim)
    
        def get_value(self, action=None, key=None):
            if key is None:
                key = self._problem.key(action)
            return float(self._q.predict(key))
    
        def choose_best_action(self, actions=None, explore=False):
            if actions is None:
                actions = self._problem.actions()
            if explore:
                epsilon = self._epsilon
            else:
                epsilon = 0.0
    
            if np.random.rand() < epsilon:
                action = actions[np.random.randint(len(actions))]
                key = self._problem.key(action)
                return action, self._q.predict(key)
            else:
                keys = [self._problem.key(action) for action in actions]
                values = self._q.predict(keys)
                best_index = self._argbest(values)
                return actions[best_index], values[best_index]
    
        def train(self, batches=None, batch_size=None, replay_count=None, iterations=None, epsilon=None):
            if batches is None:
                batches = self._batches
            if batch_size is None:
                batch_size = self._batch_size
            if replay_count is None:
                replay_count = self._replay_count
            if epsilon is None:
                self._epsilon = self._epsilon_orig
            else:
                self._epsilon = epsilon

            repk = -1
            for batch in range(batches):
                if self._q.is_trained():
                    # only update epsilon when nn is trained. This allows for initial batch to be
                    # completely random, and it allows for calling train a second time while
                    # not running another random batch.
                    self._epsilon *= self._epsilon_decay_factor
                    self._epsilon = max(self._epsilon_min, self._epsilon)

                samples = []
                replay_samples = []

                for rep in range(batch_size):
                    repk += 1
                    step = 0
                    self._problem.reset()
                    action, q_val = self.choose_best_action(explore=True)
                    key, reward = self._problem.do(action)

                    while not self._problem.is_terminal():
                        # step
                        step += 1
                        action, q_val = self.choose_best_action(explore=True)
                        samples.append([*key, reward, q_val])
                        key, reward = self._problem.do(action)
                        replay_samples.append(key)

                    # now samples has one more index, but when replay is accessed, it ignores where
                    # q_val is 0, i.e., the last row where replay_samples has no element
                    samples.append([*key, reward, 0])
                    replay_samples.append(key)  # place holder
                    if self._verbose:
                        if rep % 10 == 0 or rep == batch_size - 1:
                            report = 'batch={:d} rep={:d} epsilon={:.3f} steps={:d}'
                            print(report.format(batch, repk, self._epsilon, int(step)))

                samples = np.array(samples)
                x = samples[:, :-2]  # state-action pairs
                t = samples[:, -2:-1] + samples[:, -1:]  # reward + q_val
                self._q.fit(x, t, iterations=iterations, verbose=False)

                replay_samples = np.array(replay_samples)
                for replay in range(replay_count):
                    non_zero_q_vals = samples[:, -1] != 0
                    if sum(non_zero_q_vals) == 0:
                        break
                    recalc_q_vals = replay_samples[non_zero_q_vals, :]
                    samples[non_zero_q_vals, -1:] = self._q.predict(recalc_q_vals)
                    t = samples[:, -2:-1] + samples[:, -1:]  # reward + q_val
                    self._q.fit(x, t, iterations=iterations, verbose=False)
