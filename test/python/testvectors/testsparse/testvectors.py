"""
Sparse Vectors module tests
"""

import unittest

from txtai.vectors import SparseVectors, SparseVectorsFactory


class TestSparseVectors(unittest.TestCase):
    """
    Sparse Vectors tests.
    """

    def testCustom(self):
        """
        Test custom sparse vectors instance
        """

        self.assertIsNotNone(
            SparseVectorsFactory.create({"method": "txtai.vectors.SparseSTVectors", "path": "sparse-encoder-testing/splade-bert-tiny-nq"})
        )

    def testNotSupported(self):
        """
        Test exceptions for unsupported methods
        """

        vectors = SparseVectors(None, None, None)

        self.assertRaises(ValueError, vectors.truncate, None)
        self.assertRaises(ValueError, vectors.quantize, None)

    def testNotFound(self):
        """
        Test unresolvable vector backend
        """

        with self.assertRaises(ImportError):
            SparseVectorsFactory.create({"method": "notfound.vectors"})
