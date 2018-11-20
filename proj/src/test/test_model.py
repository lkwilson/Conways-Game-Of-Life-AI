import unittest
import numpy as np

from cgolai.cgol import Model


class TestModel(unittest.TestCase):
    def test_bool(self):
        model = Model((10, 10))
        flip = np.zeros((10, 10), dtype=bool)
        flip[1:4, 3] = True
        model.flip(flip)
        model.step()
        self.assertTrue(np.all(model.board[2, 2:4]))
        model.step()
        self.assertTrue(np.all(model.board[1:4, 3]))

    def test_board_vs_base(self):
        model = Model((10, 10))
        self.assertTrue(model.base is not model.board)
        self.assertTrue(model.base[0] is not model.board[0])

        flip = np.zeros((10, 10), dtype=bool)
        flip[1:4, 3] = True
        model.flip(flip)
        self.assertTrue(model.base is not model.board)
        self.assertTrue(model.base[0] is not model.board[0])

    def test_bool_flip(self):
        model = Model((10, 10))
        flip = np.zeros((10, 10), dtype=bool)
        flip[0:2, 3] = True
        flip[9, 3] = True
        model.flip(flip)
        model.step()
        self.assertTrue(np.all(model.board[0, 2:5]))

    def test_flip_base_const(self):
        model = Model((10, 10))

        flip = np.zeros((10, 10), dtype=bool)
        flip[0:2, 3] = True
        flip[9, 3] = True

        before_base = model.base.copy()
        model.flip(flip)
        after_base = model.base
        
        self.assertTrue(np.all(before_base == after_base))

    def test_flip_in_place(self):
        model = Model((10, 10))

        flip = np.zeros((10, 10), dtype=bool)
        flip[0:2, 3] = True
        flip[9, 3] = True

        before_base = model.base
        before_board = model.board
        model.flip(flip)
        after_base = model.base
        after_board = model.board
        
        self.assertTrue(before_base is after_base)
        self.assertTrue(before_board is after_board)

    def test_flip_map(self):
        model = Model((10, 10))

        flip = np.zeros((10, 10), dtype=bool)
        flip[0:2, 3] = True
        flip[9, 3] = True

        model.flip(flip)
        self.assertTrue(np.all(model.get_flip_map(0) == flip))

    def test_get_step(self):
        model = Model((10, 10))

        init_base = model.base_record[-1]
        init_board = model.board_record[-1]

        flip = np.zeros((10, 10), dtype=bool)
        flip[0:2, 3] = True
        flip[9, 3] = True

        model.flip(flip)

        self.assertTrue(model.get_flip(0)[0] is init_base)
        self.assertTrue(model.get_flip(0)[1] is init_board)
        model.step()
        self.assertTrue(np.all(model.get_flip(0)[0] == init_base))
        self.assertTrue(model.get_flip(0)[1] is init_board)

        mid_board = model.board
        mid_base = model.base

        model.flip(flip)
        model.step()

        self.assertTrue(model.get_flip(1)[0] is mid_base)
        self.assertTrue(model.get_flip(1)[1] is mid_board)

    def test_load_iter(self):
        model = Model((10, 10))

        flip = np.zeros((10, 10), dtype=bool)
        flip[0:2, 3] = True
        flip[9, 3] = True

        model.flip(flip)
        model.step()

        mid_board = model.board
        mid_base = model.base

        model.flip(flip)
        model.step()
        model.flip(flip)
        model.step()
        model.load_iter(1)
        self.assertTrue(model.base is mid_base)
        self.assertTrue(model.board is mid_board)

    def test_load_step_record(self):
        pass

    def test_record(self):
        model = Model((10, 10))

        flip = np.zeros((10, 10), dtype=bool)
        flip[0:2, 3] = True
        flip[9, 3] = True

        model.flip(flip)
        self.assertTrue(model.base is model.base_record[0])
        self.assertTrue(model.board is model.board_record[0])

    def test_load(self):
        pass

    def test_save(self):
        pass
