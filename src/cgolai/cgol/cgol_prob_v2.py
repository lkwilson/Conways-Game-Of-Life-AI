import numpy as np

from cgolai.ai import ProblemNN


class CgolProblem(ProblemNN):
    def __init__(self, model, init_flip=None, density=.2, high_density=.5, pop_record_size=3):
        """

        :param model: The model
        :param init_flip: The initial position
        """
        super().__init__(maximize=True)
        self._model = model
        self._init_flip = init_flip
        self.cols = self._model.size[1]
        self.rows = self._model.size[0]
        self.length = self.cols * self.rows
        self.key_dim = self.length
        self._actions = self.ohe(self.key_dim)
        self._density = density
        self._high_density = high_density
        self._pop_record = [None for _ in range(pop_record_size)]
        self._too_static = 0
        self._static_max = pop_record_size

    def is_terminal(self):
        """ Returns True iff the state is a terminal state """
        return self.too_static() or self._pop_record[-1] == 0

    def _pop(self):
        return np.sum(self._model.base)

    def key(self, action):
        """ Return the state-action key from the current state given the action """
        return [*self._model.board.reshape(self.length), *action]

    def actions(self):
        """ Returns list of possible actions """
        return self._actions

    def shift(self, new_pop):
        for i in range(len(self._pop_record)-1):
            self._pop_record[i] = self._pop_record[i+1]
        self._pop_record[-1] = new_pop

    def do(self, action):
        """ Perform the specified action on current state, and returns (state-action key, reward) """
        # before action
        key = self.key(action)

        # do action
        val = action.index(1)
        loc = (val // self.cols, val % self.rows)
        self._model.flip(loc)
        self._model.step()

        # after action
        new_pop = np.sum(self._model.base)
        if self.is_terminal():  # if all cells have died (or are static) :(
            reward = -5
        elif new_pop - self._pop_record[-1] == 0:  # if no change from previous state >:(
            reward = -1
        elif new_pop / self.length > self._high_density:  # lots of cells :))
            reward = 5
        else:  # it better have a plan
            reward = 1

        # update pop record
        if new_pop in self._pop_record:  # if no change from previous state >:(
            self._too_static += 1
        elif self._too_static > 0:
            self._too_static -= 1
        self.shift(new_pop)

        # return reward and key
        return key, reward

    def too_static(self):
        return self._too_static > self._static_max

    def get_model(self):
        return self._model

    def reset(self, init_flip=None):
        """ Initialize the state to the initial position """
        # if init_flip supplied, use it. If not, use default or generate
        # random board if no default.
        if init_flip is None:
            init_flip = self._init_flip
        if init_flip is None:
            init_flip = np.random.rand(self.rows, self.cols) < self._density

        # reset model
        self._model.clear_board()
        self._model.step()
        self._model.flip(init_flip)

        # set problem related resets
        # notice that pop is measured post flip init but before action applied
        for i, _ in enumerate(self._pop_record):
            self._pop_record[i] = None
        self._pop_record[-1] = np.sum(self._model.board)
        self._too_static = 0

    def get_key_dim(self):
        """ Returns the length of the state-action key """
        return self.length + self.key_dim
