import numpy as np

class Model:
    def __init__(self, size=None, initBoard=None, record=True, verbose=False):
        """
        Args:
            - size : (row, col) : The size of the cgol board
            - initBoard : np.array : The cgol board (to copy)
            - record : bool : If the model should record game states

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
        if initBoard is None:
            initBoard = np.zeros(size, dtype=bool)
        self.size = initBoard.shape

        self.base = initBoard.copy()
        self.board = initBoard.copy()

        self.record = record
        if self.record:
            self.index = 0
            self.baseRecord = [self.base]
            self.boardRecord = [self.board]

        self.watchers = []
        self.verbose = verbose

    def flip(self, loc):
        '''
        loc : (row, col) by matrix indexing (origin is top left)
        loc : np.array(self.size, dtype=bool)
        '''
        if isinstance(loc, tuple):
            self.board[loc] = not self.board[loc]
        else:
            self.board[:,:] = self.board != loc
        self.notify()
        if self.verbose:
            print('flipped:type(loc)', loc)

    def clearFlip(self):
        self.setBoard(self.base)

    def step(self, resetFlipBoard=True):
        # I wanted the step to be efficient. I used code from here:
        # https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
        nbrs_count = sum(np.roll(np.roll(self.board, i, 0), j, 1)
                     for i in (-1, 0, 1) for j in (-1, 0, 1)
                     if (i != 0 or j != 0))
        self.board = (nbrs_count == 3) | (self.board & (nbrs_count == 2))
        self.base = self.board.copy()

        if self.record:
            if self.index+1 < len(self.baseRecord):
                self.baseRecord[self.index] = self.base
                self.baseRecord = self.baseRecord[:self.index+1]
                self.boardRecord[self.index] = self.board
                self.boardRecord = self.boardRecord[:self.index+1]
            else:
                self.baseRecord.append(self.base)
                self.boardRecord.append(self.board)
            self.index += 1
        self.notify()
        if self.verbose:
            print('step:',self.index)

    def back(self):
        self.loadIter(self.index-1)

    def forward(self, generate=True, **args):
        """
        - generates the next step if at end and if generate is true.
        - forwards args to step if generate is True
        """
        if self.index+1>=len(self.baseRecord) and generate:
            self.step(**args)
        else:
            self.loadIter(self.index+1)

    def loadIter(self, n):
        """ Loads iteration n, before the nth step, record[n] """
        if not self.record:
            return

        if n >= len(self.baseRecord):
            n = len(self.baseRecord)-1
        elif n < 0:
            n = 0

        self.base = self.baseRecord[n]
        self.board = self.boardRecord[n]
        self.index = n
        self.notify()

    def getCell(self, row, col):
        return self.board[row,col]

    def setBoard(self, board):
        self.board[:,:] = board
        self.notify()

    def getStep(self, n):
        """ get the step from n to n+1 (board[n] --step--> base[n+1]) """
        return self.boardRecord[n], self.baseRecord[n+1]

    def getFlip(self, n):
        """ get the flip from base to board (base[n] --flips--> board[n] """
        return self.baseRecord[n], self.boardRecord[n]

    def getFlipMap(self, n=None):
        """
        get the flip matrix from base to board M where
        base[n] --flip by matrix M--> board[n]
        """
        if n is None:
            base, board = self.base, self.board
        else:
            base, board = self.getFlip(n)
        return base != board

    def save(self):
        pass # TODO

    def load(self):
        pass # TODO

    def close(self, save=False):
        if save:
            self.save()
        # TODO

    def watch(self, obj):
        self.watchers.append(obj)
    
    def notify(self):
        for watcher in self.watchers:
            watcher.update()
