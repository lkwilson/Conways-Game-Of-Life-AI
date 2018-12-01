import numpy as np

from cgolai.ai import ProblemNN


class CgolProblem(ProblemNN):
    def __init__(self, model, init_flip=None, density=.2, change_based_reward=False,
                 allow_idle=False):
        """

        :param model: The model
        :param init_flip: The initial position
        """
        super().__init__(maximize=True)
        self.model = model
        self.init_flip = init_flip
        self.cols = self.model.size[1]
        self.rows = self.model.size[0]
        self.length = self.cols * self.rows
        self._key_dim = self.length + int(allow_idle)
        self._actions = self.ohe(self._key_dim)
        self._density = density
        self._change_based_reward = change_based_reward

    def is_terminal(self):
        """ Returns True iff the state is a terminal state """
        return self._reward() == 0

    def _reward(self, previous_population=None):
        if self._change_based_reward:
            # assumed has stepped since reward comes from state transitions
            if previous_population is None:
                previous_population = np.sum(self.model.base_record[-1])
            # here, base and board for the current state are the same.
            # Using base instead will ignore outside input which modified
            # board post step (alternate state resets or external intervention).
            return np.sum(self.model.base) - previous_population
        else:
            return np.sum(self.model.base)

    def key(self, action):
        """ Return the state-action key from the current state given the action """
        return [*self.model.board.reshape(self.length), *action]

    def actions(self):
        """ Returns list of possible actions """
        return self._actions

    def do(self, action):
        """ Perform the specified action on current state, and returns (state-action key, reward) """
        key = self.key(action)
        # TODO make change_based_reward work without record
        val = action.index(1)
        if val != self.length:
            loc = (val // self.cols, val % self.rows)
            self.model.flip(loc)
        self.model.step()
        return key, self._reward()

    def reset(self):
        """ Initialize the state to the initial position """
        if self.init_flip is None:
            init_flip = np.random.rand(self.rows, self.cols) < self._density
        else:
            init_flip = self.init_flip
        self.model.clear_board()
        self.model.step()
        self.model.flip(init_flip)

    def get_key_dim(self):
        """ Returns the length of the state-action key """
        return self.length + self._key_dim
