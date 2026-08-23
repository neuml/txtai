"""
WordVectors module tests
"""

import os
import shutil
import tempfile
import unittest

from unittest.mock import patch

import numpy as np

from huggingface_hub.errors import HFValidationError
from txtai.vectors import VectorsFactory
from txtai.vectors.dense.words import create, transform, WordVectors


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

    def testIndexCheckpoint(self):
        """
        Test word vector indexing writes a checkpoint file and a second run over the same
        documents is served entirely from it
        """

        checkpoint = os.path.join(tempfile.gettempdir(), "wordvectors-checkpoint")
        shutil.rmtree(checkpoint, ignore_errors=True)

        documents = [(x, "This is a test", None) for x in range(10)]
        model = VectorsFactory.create({"path": self.path, "parallel": False}, None)

        # First run builds the checkpoint
        model.index(documents, 5, checkpoint)
        self.assertTrue(os.path.exists(os.path.join(checkpoint, model.vectorsid())))

        # Second run over the same documents must be fully served from the checkpoint
        with patch.object(WordVectors, "encode", wraps=model.encode) as encode:
            model.index(documents, 5, checkpoint)
            encode.assert_not_called()

    @patch("os.cpu_count")
    def testIndexCheckpointParallel(self, cpucount):
        """
        Test word vector parallel indexing writes a checkpoint file
        """

        # Mock CPU count
        cpucount.return_value = 1

        checkpoint = os.path.join(tempfile.gettempdir(), "wordvectors-checkpoint-parallel")
        shutil.rmtree(checkpoint, ignore_errors=True)

        documents = [(x, "This is a test", None) for x in range(10)]
        model = VectorsFactory.create({"path": self.path, "parallel": True}, None)

        model.index(documents, 5, checkpoint)
        self.assertTrue(os.path.exists(os.path.join(checkpoint, model.vectorsid())))

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

        # Test non-existent local path raises an exception
        with self.assertRaises((IOError, HFValidationError)):
            VectorsFactory.create({"method": "words", "path": os.path.join(tempfile.gettempdir(), "noexist")}, None)

        # Test non-existent hub path is handled properly
        self.assertFalse(WordVectors.ismodel("invalid/user/noexist"))

    def testTransform(self):
        """
        Test word vector transform
        """

        model = VectorsFactory.create({"path": self.path}, None)
        self.assertEqual(len(model.transform((None, ["txtai"], None))), 300)
