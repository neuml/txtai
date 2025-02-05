"""
WordVectors module tests
"""

import os
import tempfile
import unittest

from unittest.mock import patch

import numpy as np

from huggingface_hub.errors import HFValidationError
from txtai.vectors import VectorsFactory
from txtai.vectors.words import create, transform


class TestWordVectors(unittest.TestCase):
    """
    Vectors tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Sets the pretrained model to use
        """

        # Test with pretrained glove quantized vectors
        cls.path = "neuml/glove-6B-quantized"

    @patch("os.cpu_count")
    def testIndex(self, cpucount):
        """
        Test word vectors indexing
        """

        # Mock CPU count
        cpucount.return_value = 1

        # Generate data
        documents = [(x, "This is a test", None) for x in range(1000)]

        model = VectorsFactory.create({"path": self.path, "parallel": True}, None)

        ids, dimension, batches, stream = model.index(documents, 1)

        self.assertEqual(len(ids), 1000)
        self.assertEqual(dimension, 300)
        self.assertEqual(batches, 1000)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (1, 300))

    @patch("os.cpu_count")
    def testIndexBatch(self, cpucount):
        """
        Test word vectors indexing with batch size set
        """

        # Mock CPU count
        cpucount.return_value = 1

        # Generate data
        documents = [(x, "This is a test", None) for x in range(1000)]

        model = VectorsFactory.create({"path": self.path, "parallel": True}, None)

        ids, dimension, batches, stream = model.index(documents, 512)

        self.assertEqual(len(ids), 1000)
        self.assertEqual(dimension, 300)
        self.assertEqual(batches, 2)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (512, 300))
            self.assertEqual(np.load(queue).shape, (488, 300))

    def testIndexSerial(self):
        """
        Test word vector indexing in single process mode
        """

        # Generate data
        documents = [(x, "This is a test", None) for x in range(1000)]

        model = VectorsFactory.create({"path": self.path, "parallel": False}, None)

        ids, dimension, batches, stream = model.index(documents, 1)

        self.assertEqual(len(ids), 1000)
        self.assertEqual(dimension, 300)
        self.assertEqual(batches, 1000)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (1, 300))

    def testIndexSerialBatch(self):
        """
        Test word vector indexing in single process mode with batch size set
        """

        # Generate data
        documents = [(x, "This is a test", None) for x in range(1000)]

        model = VectorsFactory.create({"path": self.path, "parallel": False}, None)

        ids, dimension, batches, stream = model.index(documents, 512)

        self.assertEqual(len(ids), 1000)
        self.assertEqual(dimension, 300)
        self.assertEqual(batches, 2)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (512, 300))
            self.assertEqual(np.load(queue).shape, (488, 300))

    def testLookup(self):
        """
        Test word vector lookup
        """

        model = VectorsFactory.create({"path": self.path}, None)
        self.assertEqual(model.lookup(["txtai", "embeddings", "sentence"]).shape, (3, 300))

    def testMultiprocess(self):
        """
        Test multiprocess helper methods
        """

        create({"path": self.path}, None)

        uid, vector = transform((0, "test", None))
        self.assertEqual(uid, 0)
        self.assertEqual(vector.shape, (300,))

    def testNoExist(self):
        """
        Test loading model that doesn't exist
        """

        # Test non-existent path raises an exception
        with self.assertRaises((IOError, HFValidationError)):
            VectorsFactory.create({"method": "words", "path": os.path.join(tempfile.gettempdir(), "noexist")}, None)

    def testTransform(self):
        """
        Test word vector transform
        """

        model = VectorsFactory.create({"path": self.path}, None)
        self.assertEqual(len(model.transform((None, ["txtai"], None))), 300)
