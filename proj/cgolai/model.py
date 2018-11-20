import numpy as np


class Model:
    def __init__(self, size=None, init_board=None, record=True, verbose=False,
            filename=None):
        """
        Args:
            - size : (row, col) : The size of the cgol board
            - init_board : np.array : The cgol board (to copy)
            - record : bool : If the model should record game states
            - verbose : bool : If logging should be printed
            - filename : str : A file to load

        File IO:
            If a filename is specified, then it will be used to store
            recordings when save is called. It will also load what's in the
            file on initialization. If the file doesn't exist, then it's not
            loaded nor is it created until the first save is called.

        Recording:
            Recording records the board states in baseRecord and boardRecord.
            base0 --flips--> board0 --step--> base1 --flips--> board1 --step--> base2

            baseRecord = [base0.c, base1.c, base2.c] # .c being a copy
            boardRecord = [board0.c, board1.c, base2]

            - After a flip, baseRecord doesn't change, and boardRecord reflects
              the change.
            - After a step, copy of base is added to baseRecord, and copy of
              board is added to boardRecord
        """
        if init_board is None:
            init_board = np.zeros(size, dtype=bool)
        self.size = init_board.shape

        self.base = init_board.copy()
        self.board = init_board.copy()

        self.record = record
        if self.record:
            self.index = 0
            self.base_record = [self.base]
            self.board_record = [self.board]

        self.watchers = []
        self.verbose = verbose

    def watch(self, obj):
        self.watchers.append(obj)

    def notify(self):
        for watcher in self.watchers:
            watcher.update()

    # get
    def get_step(self, n):
        """ get the step from n to n+1 (board[n] --step--> base[n+1]) """
        return self.board_record[n], self.base_record[n + 1]

    def get_cell(self, row, col):
        return self.board[row, col]

    def get_flip(self, n):
        """ get the flip from base to board (base[n] --flips--> board[n] """
        return self.base_record[n], self.board_record[n]

    def get_flip_map(self, n=None):
        """
        get the flip matrix from base to board M where
        base[n] --flip by matrix M--> board[n]
        """
        if n is None:
            base, board = self.base, self.board
        else:
            base, board = self.get_flip(n)
        return base != board

    # set
    def set_board(self, board):
        self.board[:, :] = board
        self.notify()

    # modify
    def flip(self, loc):
        """
        loc : (row, col) by matrix indexing (origin is top left)
        loc : np.array(self.size, dtype=bool)
        """
        if isinstance(loc, tuple):
            self.board[loc] = not self.board[loc]
        else:
            self.board[:, :] = self.board != loc
        self.notify()
        if self.verbose:
            print('flipped:type(loc)', loc)

    def clear_flip(self):
        self.set_board(self.base)

    def step(self):
        # I wanted the step to be efficient. I used code from here:
        # https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
        neighbors_count = sum(
            np.roll(np.roll(self.board, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0)
        )
        self.board = (neighbors_count == 3) | (self.board & (neighbors_count == 2))
        self.base = self.board.copy()

        if self.record:
            if self.index + 1 < len(self.base_record):
                self.base_record[self.index] = self.base
                self.base_record = self.base_record[:self.index + 1]
                self.board_record[self.index] = self.board
                self.board_record = self.board_record[:self.index + 1]
            else:
                self.base_record.append(self.base)
                self.board_record.append(self.board)
            self.index += 1
        self.notify()
        if self.verbose:
            print('step:', self.index)

    def back(self):
        self.load_iter(self.index - 1)

    def forward(self, generate=True):
        """
        - generates the next step if at end and if generate is true.
        - forwards args to step if generate is True
        """
        if self.index + 1 >= len(self.base_record) and generate:
            self.step()
        else:
            self.load_iter(self.index + 1)

    def load_iter(self, n):
        """ Loads iteration n, before the nth step, record[n] """
        if not self.record:
            return

        if n >= len(self.base_record):
            n = len(self.base_record) - 1
        elif n < 0:
            n = 0

        self.base = self.base_record[n]
        self.board = self.board_record[n]
        self.index = n
        self.notify()

    def clear_board(self):
        self.set_board(False)

    def invert(self):
        self.flip(np.ones(self.size, dtype=bool))

    def save(self, filename):
        pass  # TODO

    def load(self, filename):
        pass  # TODO

    def close(self, save=False):
        if save:
            self.save()
        # TODO
