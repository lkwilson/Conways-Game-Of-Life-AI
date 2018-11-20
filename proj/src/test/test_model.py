import unittest
import numpy as np
from cgolai.cgol import Model

class testTest(unittest.TestCase):
    def testBool(self):
        model = Model((10, 10))
        flip = np.zeros((10, 10), dtype=bool)
        flip[1:4,3] = True
        model.flip(flip)
        model.step()
        self.assertTrue(np.all(model.board[2,2:4]))
        model.step()
        self.assertTrue(np.all(model.board[1:4,3]))

    def testBoardVsBase(self):
        model = Model((10, 10))
        self.assertTrue(model.base is not model.board)
        self.assertTrue(model.base[0] is not model.board[0])

        flip = np.zeros((10, 10), dtype=bool)
        flip[1:4,3] = True
        model.flip(flip)
        self.assertTrue(model.base is not model.board)
        self.assertTrue(model.base[0] is not model.board[0])

    def testBool(self):
        model = Model((10, 10))
        #model.board = np.zeros((10, 10), dtype=bool)
        flip = np.zeros((10, 10), dtype=bool)
        flip[0:2,3] = True
        flip[9,3] = True
        model.flip(flip)
        model.step()
        self.assertTrue(np.all(model.board[0,2:5]))

    def testFlipBaseConst(self):
        model = Model((10, 10))

        flip = np.zeros((10, 10), dtype=bool)
        flip[0:2,3] = True
        flip[9,3] = True

        beforeBase = model.base.copy()
        model.flip(flip)
        afterBase = model.base
        
        self.assertTrue(np.all(beforeBase == afterBase))

    def testFlipInPlace(self):
        model = Model((10, 10))

        flip = np.zeros((10, 10), dtype=bool)
        flip[0:2,3] = True
        flip[9,3] = True

        beforeBase = model.base
        beforeBoard = model.board
        model.flip(flip)
        afterBase = model.base
        afterBoard = model.board
        
        self.assertTrue(beforeBase is afterBase)
        self.assertTrue(beforeBoard is afterBoard)

    def testFlipMap(self):
        model = Model((10, 10))

        flip = np.zeros((10, 10), dtype=bool)
        flip[0:2,3] = True
        flip[9,3] = True

        model.flip(flip)
        self.assertTrue(np.all(model.get_flip_map(0) == flip))

    def testGetStep(self):
        model = Model((10, 10))

        initBase = model.base_record[-1]
        initBoard = model.board_record[-1]

        flip = np.zeros((10, 10), dtype=bool)
        flip[0:2,3] = True
        flip[9,3] = True

        model.flip(flip)

        self.assertTrue(model.get_flip(0)[0] is initBase)
        self.assertTrue(model.get_flip(0)[1] is initBoard)
        model.step()
        self.assertTrue(np.all(model.get_flip(0)[0] == initBase))
        self.assertTrue(model.get_flip(0)[1] is initBoard)

        midBoard = model.board
        midBase = model.base

        model.flip(flip)
        model.step()

        self.assertTrue(model.get_flip(1)[0] is midBase)
        self.assertTrue(model.get_flip(1)[1] is midBoard)

    def testLoadIter(self):
        model = Model((10, 10))

        initBoard = model.board
        initBase = model.base

        flip = np.zeros((10, 10), dtype=bool)
        flip[0:2,3] = True
        flip[9,3] = True

        model.flip(flip)
        model.step()

        midBoard = model.board
        midBase = model.base

        model.flip(flip)
        model.step()
        model.flip(flip)
        model.step()
        model.load_iter(1)
        self.assertTrue(model.base is midBase)
        self.assertTrue(model.board is midBoard)

    def testLoadStepRecord(self):
        pass

    def testRecord(self):
        model = Model((10, 10))

        flip = np.zeros((10, 10), dtype=bool)
        flip[0:2,3] = True
        flip[9,3] = True

        model.flip(flip)
        self.assertTrue(model.base is model.base_record[0])
        self.assertTrue(model.board is model.board_record[0])
