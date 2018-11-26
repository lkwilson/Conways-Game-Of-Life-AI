class Problem:
    def __init__(self, maximize=True):
        """ problem should be initialized to the point of where actions will return actions """
        self.maximize = maximize

    def is_terminal(self):
        """ Returns if the state is a terminal state """
        pass

    def actions(self):
        """ Returns possible actions given the current state """
        pass

    def get_key_dim(self):
        """ Return the dimensions of the key """
        return len(self.key(self.actions()[0])[0])

    def key(self, action):
        """ Return the state-action key from the current state given the action """
        pass

    def do(self, action):
        """ Perform the specified action on current state.

        Returns the (old_state, action) key
        """
        pass

    def reset(self):
        """ Initialize the state to the initial position """
        pass

