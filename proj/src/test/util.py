from cgolai.ai import Problem


class Hanoi(Problem):
    def __init__(self, size=3):
        super().__init__(maximize=False)
        self.size = size
        self.state = None
        self.reward = 1
        self.reset()

    def is_terminal(self):
        """ Returns if the state is a terminal state """
        return len(self.state[0]) == 0 and len(self.state[1]) == 0

    def actions(self):
        """ Returns possible actions given the current state """
        ret = []
        for srci, src in enumerate(self.state):
            for dsti, dst in enumerate(self.state):
                has_src = len(src) != 0  # src peg has a piece
                same_peg = srci == dsti  # dst peg is same peg
                no_dst = len(dst) == 0  # dst peg has no pieces
                if has_src and not same_peg and (no_dst or src[0] < dst[0]):
                    ret.append([srci, dsti])
        return ret

    def reset(self):
        """ Initialize the state to the initial position """
        peg1 = list(range(1, self.size + 1))
        self.state = [peg1, [], []]

    def key(self, action):
        """ Return the state-action key from the current state given the action """
        return (*self._state_rep(),  *action)

    def get_key_dim(self):
        return 5

    def print_board(self):
        lens = [len(p) for p in self.state]
        for height in range(max(lens), 0, -1):
            row = ""
            for p in range(3):
                if lens[p] >= height:
                    row += str(self.state[p][lens[p] - height]) + ' '
                else:
                    row += '  '
            print(row)
        print('------')
        print()

    def do(self, action):
        """ Perform the specified action on current state.

        Returns the (old_state, action) key
        """
        old_state_action_key = self.key(action)
        src, dst = action
        self.state[dst].insert(0, self.state[src].pop(0))
        return old_state_action_key, self.reward

    def _state_rep(self):
        newrep = list(range(self.size))
        for pegi, peglist in enumerate(self.state):
            for disk in peglist:
                newrep[disk - 1] = pegi + 1
        return newrep


class HanoiNN(Hanoi):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def key(self, action):
        """ Return the state-action key from the current state given the action """
        return [*self._state_rep(),  *action]
