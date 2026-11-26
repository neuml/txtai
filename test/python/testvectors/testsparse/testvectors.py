"""
Sparse Vectors module tests
"""

import unittest

from unittest.mock import patch

from scipy.sparse import csr_matrix, vstack

from txtai.vectors import SparseVectors, SparseVectorsFactory


class InvalidVectors:
    """
    Invalid vectors class
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None


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

    def testDefaultNormalize(self):
        """
        Test defaultnormalize method
        """

        vectors = SparseVectors(None, None, None)
        self.assertFalse(vectors.defaultnormalize())

    def testInvalid(self):
        """
        Test invalid vector backend
        """

        with self.assertRaises(ImportError):
            SparseVectorsFactory.create({"method": "testvectors.testsparse.testvectors.InvalidVectors"})

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

    def testVectors(self):
        """
        Test empty, single and multiple batches when rebuilding sparse vectors
        """

        with patch.object(SparseVectors, "loadmodel", return_value=None):
            vectors = SparseVectors({"method": "custom"}, None, None)

        documents = [(i, str(i + 1), None) for i in range(5)]
        with patch.object(vectors, "vectorize", side_effect=lambda rows, _: csr_matrix([[int(row), 0] for row in rows])):
            for size in (0, 1, 5):
                with self.subTest(size=size), patch("txtai.vectors.sparse.base.vstack", wraps=vstack) as stack:
                    ids, dimensions, matrix = vectors.vectors(documents[:size], batchsize=2)
                    self.assertEqual(ids, list(range(size)))
                    if size:
                        self.assertEqual(dimensions, 2)
                        self.assertEqual(matrix.toarray().tolist(), [[i + 1, 0] for i in range(size)])
                    else:
                        self.assertIsNone(matrix)
                    self.assertLessEqual(stack.call_count, 1)
