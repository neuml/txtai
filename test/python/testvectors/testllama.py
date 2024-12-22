"""
Llama module tests
"""

import os
import unittest

import numpy as np

from txtai.vectors import VectorsFactory


class TestLlamaCpp(unittest.TestCase):
    """
    llama.cpp vectors tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Create LlamaCpp instance.
        """

        cls.model = VectorsFactory.create({"path": "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q2_K.gguf"}, None)

    def testIndex(self):
        """
        Test indexing with LlamaCpp vectors
        """

        ids, dimension, batches, stream = self.model.index([(0, "test", None)])

        self.assertEqual(len(ids), 1)
        self.assertEqual(dimension, 768)
        self.assertEqual(batches, 1)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (1, 768))
