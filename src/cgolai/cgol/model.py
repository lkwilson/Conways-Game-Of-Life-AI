import os
import gzip
import pickle
import numpy as np


class Model:
    def __init__(self, size=None, init_board=None, record=True, verbose=False,
                 filename=None, load=False, watchers=None, mutate_density=0.5):
        """
        Constructing:
            Initially, the model is inactive.
            - If load is True, the model is loaded from filename.
            - If the load fails or if load is False, a build is attempted. A
              build requires init_board or size be specified.
            - If the model isn't loaded or built, then the model is left
              inactive.
            - If the model is inactive, then base, board, base_record,
              board_record, index, and size are None, and active is False.

        Args:
            - size : (row, col) : The size of the cgol board
            - init_board : np.array : The cgol board (to copy)
            - record : bool : If the model should record game states
            - verbose : bool : If logging should be printed
            - filename : str : A file to load
            - watchers : list-like : A list of objects who watch model

        Members:
            - active : True iff the model is active
            - base : The state
            - base_record : A record of bases
            - board : The state with flip modifications
            - board_record : A record of boards
            - filename : The file used to store the model
            - index : The current index in record
            - record : If the state is being recorded
            - size : The size of the cgol board
            - verbose : If logging is printed to stdout
            - watchers : Watchers of the model's state

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
        self.record = record
        self.verbose = verbose

        self.base = None
        self.board = None
        self.base_record = None
        self.board_record = None

        self.index = None
        self.size = None

        self.filename = filename
        self.watchers = []
        self.active = False

        self._mutate_density = mutate_density

        if not load or not self.load():
            self.build(init_board, size)

        if watchers is not None:
            for watcher in watchers:
                self.watch(watcher)

    def build(self, init_board, size):
        if init_board is None:
            if size is None:
                return
            init_board = np.zeros(size, dtype=bool)
        self.size = init_board.shape
        self.base = init_board.copy()
        self.board = init_board.copy()
        if self.record:
            self.index = 0
            self.base_record = [self.base]
            self.board_record = [self.board]
        self.active = True

    def assert_active(self):
        if not self.active:
            raise Exception("model not active")

    def watch(self, obj):
        self.watchers.append(obj)

    def notify(self):
        for watcher in self.watchers:
            watcher.update()

    # get
    def get_step(self, n):
        """ get the step from n to n+1 (board[n] --step--> base[n+1]) """
        self.assert_active()
        return self.board_record[n], self.base_record[n + 1]

    def get_cell(self, row, col):
        self.assert_active()
        return self.board[row, col]

    def get_flip(self, n=None):
        """ get the flip from base to board (base[n] --flips--> board[n] """
        self.assert_active()
        if n is None:
            return self.base, self.board
        return self.base_record[n], self.board_record[n]

    def get_flip_map(self, n=None):
        """
        get the flip matrix from base to board M where
        base[n] --flip by matrix M--> board[n]
        """
        self.assert_active()
        base, board = self.get_flip(n)
        return base != board

    # set
    def set_board(self, board):
        """
        This sets self.board to the contents of board. The self.board pointer
        doesn't change, and it assigns using numpy, so broadcasting works.
        (i.e., board=False sets all cells to False)
        """
        self.assert_active()
        self.board[:, :] = board
        self.notify()

    def set_filename(self, filename):
        self.filename = filename

    # modify
    def flip(self, loc):
        """
        loc : (row, col) by matrix indexing (origin is top left)
        loc : np.array(self.size, dtype=bool)
        """
        self.assert_active()
        if isinstance(loc, tuple):
            self.board[loc] = not self.board[loc]
        else:
            self.board[:, :] = self.board != loc
        self.notify()
        if self.verbose:
            print('flipped:type(loc)', loc)

    def clear_flip(self):
        self.assert_active()
        self.set_board(self.base)

    def step(self):
        self.assert_active()
        # I wanted the step to be efficient. I used code from here:
        # https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
        neighbors_count = sum(
            np.roll(np.roll(self.board, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0)
        )
        self.board = (neighbors_count == 3) | (self.board & (neighbors_count == 2))
        # end cite
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
        self.assert_active()
        self.load_iter(self.index - 1)

    def forward(self, generate=True):
        """
        - generates the next step if at end and if generate is true.
        - forwards args to step if generate is True
        """
        self.assert_active()
        if self.index + 1 >= len(self.base_record) and generate:
            self.step()
        else:
            self.load_iter(self.index + 1)

    def load_iter(self, n=None):
        """ Loads iteration n, before the nth step, record[n] """
        self.assert_active()
        if not self.record:
            return

        if n is None or n >= len(self.base_record):
            n = len(self.base_record)-1
        elif n < 0:
            n = 0

        self.base = self.base_record[n]
        self.board = self.board_record[n]
        self.index = n
        self.notify()

    def clear_board(self):
        self.assert_active()
        self.set_board(False)

    def invert(self):
        self.assert_active()
        self.flip(np.ones(self.size, dtype=bool))

    def mutate(self):
        """ Apply random flip map """
        self.flip(np.random.rand(self.size) < self._mutate_density)

    # io
    def presave(self):
        filename = self.filename
        self.filename = None
        watchers = self.watchers
        self.watchers = None
        return filename, watchers

    def postsave(self, state):
        self.filename, self.watchers = state

    def save(self):
        if self.filename is not None:
            filename = self.filename
            state = self.presave()
            with open(filename, 'wb+') as f:
                # mode is wb+, and GzipFile can't auto resolve, so mode has to be given
                fc = gzip.GzipFile(fileobj=f, mode='wb')
                pickle.dump(self, fc)
                fc.close()
            self.postsave(state)
            return True
        return False

    def load(self):
        """
        Loads state from self.filename if not None and if file exists.
        self.filename and self.watchers are preserved.

        Returns:
            True if loaded
        """
        if self.filename is not None and os.path.exists(self.filename):
            with gzip.open(self.filename) as f:
                self.load_state(pickle.load(f))
            return True
        return False

    def load_state(self, state):
        self.active = state.active
        self.size = state.size
        self.record = state.record
        self.index = state.index
        self.base_record = state.base_record
        self.board_record = state.board_record
        self.base = state.base
        self.board = state.board

        # this will fix self.base and self.board pointers
        if self.active:
            self.load_iter(self.index)
