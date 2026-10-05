"""
Sparse Sentence Transformers module tests
"""

import os
import tempfile
import unittest

from unittest.mock import patch

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

    def testVectors(self):
        """
        Test building vectors with sentence-transformers vectors
        """

        model = SparseVectorsFactory.create({"method": "sentence-transformers", "path": "sparse-encoder-testing/splade-bert-tiny-nq"})

        # Spool into an empty directory to check the temporary file doesn't outlive the call
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(tempfile, "tempdir", directory):
                ids, dimension, embeddings = model.vectors([(0, "test", None)])

            self.assertEqual(len(ids), 1)
            self.assertEqual(dimension, 30522)
            self.assertEqual(embeddings.shape, (1, 30522))
            self.assertEqual(os.listdir(directory), [])
