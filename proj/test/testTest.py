import unittest
import cgolai

class testTest(unittest.TestCase):
    def setUp(self):
        self.word = 'word'

    def testEqual(self):
        self.assertEqual(self.word, 'word')

    def testBool(self):
        self.assertTrue(True)
        self.assertFalse(False)

    def testEqp(self):
        a = None
        with self.assertRaises(AttributeError):
            a.badMethod()

    def tearDown(self):
        self.word = None
