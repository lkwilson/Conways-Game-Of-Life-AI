import os
import unittest
import numpy as np

from cgolai.cgol import Model


class TestModel(unittest.TestCase):
    def setUp(self):
        self.filename = os.path.join('src', 'test', 'test.dat')
        self.model = Model(size=(10, 10))
        self.model2 = Model(size=(10, 10))
        self.flip = np.zeros((10, 10), dtype=bool)

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_bool(self):
        flip = np.zeros((10, 10), dtype=bool)
        flip[1:4, 3] = True
        self.model.flip(flip)
        self.model.step()
        self.assertTrue(np.all(self.model.board[2, 2:4]))
        self.model.step()
        self.assertTrue(np.all(self.model.board[1:4, 3]))

    def test_board_vs_base(self):
        self.assertTrue(self.model.base is not self.model.board)
        self.assertTrue(self.model.base[0] is not self.model.board[0])

        self.model.flip(self.flip)
        self.assertTrue(self.model.base is not self.model.board)
        self.assertTrue(self.model.base[0] is not self.model.board[0])

    def test_bool_flip(self):
        flip = np.zeros((10, 10), dtype=bool)
        flip[0:2, 3] = True
        flip[9, 3] = True
        self.model.flip(flip)
        self.model.step()
        self.assertTrue(np.all(self.model.board[0, 2:5]))

    def test_flip_base_const(self):
        before_base = self.model.base.copy()
        self.model.flip(self.flip)
        after_base = self.model.base
        
        self.assertTrue(np.all(before_base == after_base))

    def test_flip_in_place(self):
        before_base = self.model.base
        before_board = self.model.board
        self.model.flip(self.flip)
        after_base = self.model.base
        after_board = self.model.board
        
        self.assertTrue(before_base is after_base)
        self.assertTrue(before_board is after_board)

    def test_flip_map(self):
        self.model.flip(self.flip)
        self.assertTrue(np.all(self.model.get_flip_map(0) == self.flip))

    def test_get_step(self):
        init_base = self.model.base_record[-1]
        init_board = self.model.board_record[-1]

        self.model.flip(self.flip)

        self.assertTrue(self.model.get_flip(0)[0] is init_base)
        self.assertTrue(self.model.get_flip(0)[1] is init_board)
        self.model.step()
        self.assertTrue(np.all(self.model.get_flip(0)[0] == init_base))
        self.assertTrue(self.model.get_flip(0)[1] is init_board)

        mid_board = self.model.board
        mid_base = self.model.base

        self.model.flip(self.flip)
        self.model.step()

        self.assertTrue(self.model.get_flip(1)[0] is mid_base)
        self.assertTrue(self.model.get_flip(1)[1] is mid_board)

    def test_load_iter(self):
        self.model.flip(self.flip)
        self.model.step()

        mid_board = self.model.board
        mid_base = self.model.base

        self.model.flip(self.flip)
        self.model.step()
        self.model.flip(self.flip)
        self.model.step()
        self.model.load_iter(1)
        self.assertTrue(self.model.base is mid_base)
        self.assertTrue(self.model.board is mid_board)

    def test_set_filename(self):
        self.assertFalse(Model(filename=self.filename).filename is None, "filename is actually set")
        self.assertEqual(Model(filename=self.filename).filename, self.filename)

    def test_record(self):
        self.model.flip(self.flip)
        self.assertTrue(self.model.base is self.model.base_record[0])
        self.assertTrue(self.model.board is self.model.board_record[0])

    def test_save_load(self):
        model_expected = Model(size=(10, 10), filename=self.filename)
        model_expected.flip(self.flip)
        model_expected.save()

        model = Model(filename=self.filename, load=True)
        self.assertTrue(np.all(model_expected.board == model.board))
        self.assertTrue(np.all(model_expected.base == model.base))

    def test_set_file(self):
        expected_base, expected_board = self.model.get_flip()
        self.model.set_filename(self.filename)
        self.model.flip(self.flip)
        self.model.save()
        self.model.load()
        self.assertTrue(np.all(expected_base == self.model.base))
        self.assertTrue(np.all(expected_board == self.model.board))

    def test_save_load_pointer(self):
        model_expected = Model((10, 10), filename=self.filename)
        model_expected.flip(self.flip)
        model_expected.save()

        model = Model(filename=self.filename, load=True)
        self.assertTrue(model.base is model.get_flip()[0])
        self.assertTrue(model.board is model.get_flip()[1])

    def test_double_save_load(self):
        model_expected = Model(size=(10, 10), filename=self.filename)
        model_expected.flip(self.flip)
        model_expected.save()
        model_expected.save()

        self.model = Model(filename=self.filename, load=True)
        self.assertTrue(np.all(model_expected.board == self.model.board))
        self.assertTrue(np.all(model_expected.base == self.model.base))

    def test_verbose(self):
        self.model.verbose = True
        self.model.set_filename(self.filename)
        old_verbose = self.model.verbose
        self.model.save()
        self.model.load()
        self.assertEqual(self.model.verbose, old_verbose)

