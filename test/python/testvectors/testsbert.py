"""
Sentence Transformers module tests
"""

import os
import unittest

from unittest.mock import patch

import numpy as np

from txtai.vectors import VectorsFactory


class TestSTVectors(unittest.TestCase):
    """
    STVectors tests
    """

    def testIndex(self):
        """
        Test indexing with sentence-transformers vectors
        """

        model = VectorsFactory.create({"method": "sentence-transformers", "path": "paraphrase-MiniLM-L3-v2"}, None)
        ids, dimension, batches, stream = model.index([(0, "test", None)])

        self.assertEqual(len(ids), 1)
        self.assertEqual(dimension, 384)
        self.assertEqual(batches, 1)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (1, 384))

    @patch("torch.cuda.device_count")
    def testMultiGPU(self, count):
        """
        Test multiple gpu encoding
        """

        # Mock accelerator count
        count.return_value = 2

        model = VectorsFactory.create({"method": "sentence-transformers", "path": "paraphrase-MiniLM-L3-v2", "gpu": "all"}, None)
        ids, dimension, batches, stream = model.index([(0, "test", None)])

        self.assertEqual(len(ids), 1)
        self.assertEqual(dimension, 384)
        self.assertEqual(batches, 1)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (1, 384))

        # Close the multiprocessing pool
        model.close()
