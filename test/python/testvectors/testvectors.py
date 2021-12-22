"""
Vectors module tests
"""

import unittest

from txtai.vectors import Vectors


class TestVectors(unittest.TestCase):
    """
    Vectors tests.
    """

    def testNotImplemented(self):
        """
        Tests exceptions for non-implemented methods
        """

        vectors = Vectors(None, None)

        self.assertRaises(NotImplementedError, vectors.load, None)
        self.assertRaises(NotImplementedError, vectors.index, None)
        self.assertRaises(NotImplementedError, vectors.transform, None)
