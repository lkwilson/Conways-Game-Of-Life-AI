import numpy as np
from . import NNTorch as NN


class RL:
    def __init__(self, problem, shape=None,
                 verbose=False,
                 epsilon_decay_factor=0.99,
                 epsilon_init=1.0,
                 epsilon_min=0.1,
                 stochastic=False,
                 mu=0.01,
                 discount_factor=1.0,
                 batches=80,
                 batch_size=20,
                 max_steps=None,
                 max_steps_reward=None,
                 replay_count=0,
                 h=None,
                 optim=None,
                 filename=None,
                 load=False):
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
        self._max_steps = max_steps
        self._max_steps_reward = max_steps_reward
        self._replay_count = replay_count
        self._filename = filename
        self._q = None

        if load:
            self.load()
        else:
            if shape[0] is None or shape[-1] is None:
                shape[0] = self._problem.get_key_dim()
                shape[-1] = 1
            self._q = NN(shape=shape, mu=mu, h=h, optim=optim, filename=filename, load=False)

    def get_problem(self):
        return self._problem

    def choose_best_action(self, actions=None, explore=False, epsilon=None, verbose=False):
        if actions is None:
            actions = self._problem.actions()
        if not explore:
            epsilon = 0.0
        elif epsilon is None:
            epsilon = self._epsilon

        if np.random.rand() < epsilon:
            action = actions[np.random.randint(len(actions))]
            key = self._problem.key(action)
            return action, self._q.predict(key)
        else:
            keys = [self._problem.key(action) for action in actions]
            values = self._q.predict(keys)
            best_index = self._argbest(values)
            if verbose:
                print(actions, values, best_index)
            return actions[best_index], values[best_index]

    def save(self, filename=None):
        if filename is None:
            filename = self._filename
        if filename is not None:
            self._q.save(filename)

    def load(self, filename=None):
        if filename is None:
            filename = self._filename
        if filename is not None:
            if self._q is None:
                self._q = NN(filename=self._filename, load=True)
            else:
                self._q.load(filename)

    def train(self, batches=None, batch_size=None, max_steps=None, max_steps_reward=None,
              replay_count=None, iterations=None, epsilon=None):
        if batches is None:
            batches = self._batches
        if batch_size is None:
            batch_size = self._batch_size
        if replay_count is None:
            replay_count = self._replay_count
        if max_steps is None:
            max_steps = self._max_steps
        if max_steps_reward is None:
            max_steps_reward = self._max_steps_reward
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
                steps = 0
                self._problem.reset()
                action, q_val = self.choose_best_action(explore=True)
                key, reward = self._problem.do(action)

                while not self._problem.is_terminal():
                    if max_steps is not None and steps >= max_steps:
                        # the algorithm equates this to finding the goal. Better is to ensure
                        # that the problem can reach a terminal state, and then provide more accurate
                        # feedback through the reward. For example, using max_steps makes the reward
                        # whatever step happened to be last. If you manually induce terminal state after
                        # some number of steps, you can provide a reward more representative of the cutoff
                        if max_steps_reward is not None:
                            reward = max_steps_reward
                        break
                    # step
                    steps += 1
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
                        print(report.format(batch, repk, self._epsilon, int(steps)))

            #if not isinstance(samples, np.ndarray):
                #print(samples)
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
