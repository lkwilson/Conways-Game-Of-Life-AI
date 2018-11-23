import numpy as np

from ..ai import Problem


# TODO: test this?
class CGolProblem(Problem):
    def __init__(self, model, init_flip):
        self.model = model
        self.init_flip = init_flip
        self.cols = self.model.size[1]
        self.rows = self.model.size[0]
        self.length = self.cols * self.rows

    def is_terminal(self):
        """ Returns if the model is a terminal model """
        return np.sum(self.model.board) == 0

    def actions(self):
        """ Returns possible actions given the current model """
        return range(self.length+1)

    def do(self, action):
        """ Perform the specified action on current model.

        Returns the (old_model, action) key
        """
        if action != self.length:
            self.flip((action % self.cols, action % self.rows))
        self.model.step()

    def reset(self):
        """ Initialize the model to the initial position """
        # assumes is_terminal state is true for efficiency
        self.model.flip(self.init_flip)
