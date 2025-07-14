"""
Huggingface module tests
"""

import os
import unittest

import numpy as np

from txtai.vectors import VectorsFactory


class TestHFVectors(unittest.TestCase):
    """
    HFVectors tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Create HFVectors instance.
        """

        cls.model = VectorsFactory.create({"path": "sentence-transformers/nli-mpnet-base-v2"}, None)

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

    def testText(self):
        """
        Test transformers text conversion
        """

        self.model.tokenize = True
        self.assertEqual(self.model.prepare("Y 123 This is a test!"), "test")
        self.assertEqual(self.model.prepare(["This", "is", "a", "test"]), "This is a test")

        self.model.tokenize = False
        self.assertEqual(self.model.prepare("Y 123 This is a test!"), "Y 123 This is a test!")
        self.assertEqual(self.model.prepare(["This", "is", "a", "test"]), "This is a test")

    def testTransform(self):
        """
        Test transformers transform
        """

        # Sample documents: one where tokenizer changes text and one with no changes to text
        documents = [(0, "This is a test and has no tokenization", None), (1, "test tokenization", None)]

        # Run with tokenization enabled
        self.model.tokenize = True
        embeddings1 = [self.model.transform(d) for d in documents]

        # Run with tokenization disabled
        self.model.tokenize = False
        embeddings2 = [self.model.transform(d) for d in documents]

        self.assertFalse(np.array_equal(embeddings1[0], embeddings2[0]))
        self.assertTrue(np.array_equal(embeddings1[1], embeddings2[1]))

    def testTransformArray(self):
        """
        Test transformers skips transforming NumPy arrays
        """

        # Generate data and run through vector model
        data1 = np.random.rand(5, 5).astype(np.float32)
        data2 = self.model.transform((0, data1, None))

        # Test transform method returns original data
        self.assertTrue(np.array_equal(data1, data2))

    def testTransformLong(self):
        """
        Test transformers transform on long text
        """

        # Sample documents: short text and longer text
        documents = [(0, "This is long text " * 512, None), (1, "This is short text", None)]

        # Run transform and ensure it completes without errors
        embeddings = [self.model.transform(d) for d in documents]
        self.assertIsNotNone(embeddings)
