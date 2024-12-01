"""
Sentence Transformers module tests
"""

import os
import unittest

import numpy as np

from txtai.vectors import VectorsFactory


class TestSTVectors(unittest.TestCase):
    """
    STVectors tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Create STVectors instance.
        """

        cls.model = VectorsFactory.create({"method": "sentence-transformers", "path": "paraphrase-MiniLM-L3-v2"}, None)

    def testIndex(self):
        """
        Test indexing with sentence-transformers vectors.
        """

        ids, dimension, batches, stream = self.model.index([(0, "test", None)])

        self.assertEqual(len(ids), 1)
        self.assertEqual(dimension, 384)
        self.assertEqual(batches, 1)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (1, 384))
