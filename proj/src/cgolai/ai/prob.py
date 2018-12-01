from abc import ABC, abstractmethod


class Problem(ABC):
    def __init__(self, maximize=True):
        """ problem should be initialized to the point of where actions will return actions """
        self.maximize = maximize

    @abstractmethod
    def is_terminal(self):
        """ Returns True iff the state is a terminal state """
        pass

    @abstractmethod
    def actions(self):
        """ Returns list of possible actions """
        pass

    @abstractmethod
    def key(self, action):
        """ Return the state-action key from the current state given the action """
        pass

    @abstractmethod
    def do(self, action):
        """ Perform the specified action on current state, and returns (state-action key, reward) """
        pass

    @abstractmethod
    def reset(self):
        """ Initialize the state to the initial position """
        pass


class ProblemNN(Problem, ABC):
    @abstractmethod
    def get_key_dim(self):
        """ Returns the length of the state-action key """
        pass
