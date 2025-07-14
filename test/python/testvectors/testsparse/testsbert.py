"""
Sparse Sentence Transformers module tests
"""

import os
import unittest

from txtai.vectors import SparseVectorsFactory
from txtai.util import SparseArray


class TestSparseSTVectors(unittest.TestCase):
    """
    SparseSTVectors tests
    """

    def testIndex(self):
        """
        Test indexing with sentence-transformers vectors
        """

        model = SparseVectorsFactory.create({"method": "sentence-transformers", "path": "sparse-encoder-testing/splade-bert-tiny-nq"})
        ids, dimension, batches, stream = model.index([(0, "test", None)])

        self.assertEqual(len(ids), 1)
        self.assertEqual(dimension, 30522)
        self.assertEqual(batches, 1)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(SparseArray().load(queue).shape, (1, 30522))
