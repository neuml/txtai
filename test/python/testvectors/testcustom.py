"""
Custom module tests
"""

import os
import unittest

import numpy as np

from txtai.vectors import VectorsFactory


class TestCustom(unittest.TestCase):
    """
    Custom vectors tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Create custom vectors instance.
        """

        cls.model = VectorsFactory.create({"method": "txtai.vectors.HFVectors", "path": "sentence-transformers/nli-mpnet-base-v2"}, None)

    def testIndex(self):
        """
        Test transformers indexing
        """

        # Generate enough volume to test batching
        documents = [(x, "This is a test", None) for x in range(1000)]

        ids, dimension, batches, stream = self.model.index(documents)

        self.assertEqual(len(ids), 1000)
        self.assertEqual(dimension, 768)
        self.assertEqual(batches, 2)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (500, 768))

    def testNotFound(self):
        """
        Test unresolvable vector backend
        """

        with self.assertRaises(ImportError):
            VectorsFactory.create({"method": "notfound.vectors"})
