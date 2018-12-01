import numpy as np

from cgolai.ai import ProblemNN


class CgolProblem(ProblemNN):
    def __init__(self, model, init_flip=None, density=.2):
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
        self._actions = self.ohe(self.length+1)
        self._density = density

    def is_terminal(self):
        """ Returns True iff the state is a terminal state """
        return self._reward() == 0

    def _reward(self):
        return np.sum(self.model.board)

    def key(self, action):
        """ Return the state-action key from the current state given the action """
        return [*self.model.board.reshape(self.length), *action]

    def actions(self):
        """ Returns list of possible actions """
        return self._actions

    def do(self, action):
        """ Perform the specified action on current state, and returns (state-action key, reward) """
        key = self.key(action)
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
        return 2 * self.length + 1
