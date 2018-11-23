class Problem:
    def __init__(self):
        pass

    def is_terminal(self):
        """ Returns if the state is a terminal state """
        pass

    def actions(self):
        """ Returns possible actions given the current state """
        pass

    def do(self, action):
        """ Perform the specified action on current state.

        Returns the (old_state, action) key
        """
        pass

    def reset(self):
        """ Initialize the state to the initial position """
        pass

