"""
LiteRT module tests
"""

import os
import unittest

import numpy as np

from txtai.vectors import VectorsFactory


class TestLiteRT(unittest.TestCase):
    """
    LiteRT vectors tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Create LiteRT instance.
        """

        cls.model = VectorsFactory.create({"path": "neuml/bert-hash-nano-embeddings-litert/bert-hash-nano-embeddings-int4.tflite"}, None)

    def testIndex(self):
        """
        Test indexing with LiteRT vectors
        """

        ids, dimension, batches, stream = self.model.index([(0, "test", None)])

        self.assertEqual(len(ids), 1)
        self.assertEqual(dimension, 128)
        self.assertEqual(batches, 1)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (1, 128))
