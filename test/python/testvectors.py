"""
Vectors module tests
"""

import os
import tempfile
import unittest

from txtai.vectors import WordVectors, Vectors

class TestVectors(unittest.TestCase):
    """
    Vectors tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Test a WordVectors build
        """

        # Word vectors path
        path = os.path.join(tempfile.gettempdir(), "vectors")

        # Save model path
        cls.path = path + ".magnitude"

        # Build word vectors
        WordVectors.build("README.md", 300, 3, path)

    def testLoad(self):
        """
        Test loading word vectors
        """

        # Test loading model
        model = Vectors.create("words", self.path, True, None)
        self.assertEqual(len(model.transform((None, ["txtai"], None))), 300)

        # Test non-existent path raises an exception
        with self.assertRaises(IOError):
            Vectors.create("words", "/tmp/noexist", True, None)

    def testLookup(self):
        """
        Test word vector lookup
        """

        model = Vectors.create("words", self.path, True, None)
        self.assertEqual(model.lookup(["txtai", "embeddings", "sentence"]).shape, (3, 300))

    def testTransformers(self):
        """
        Test transformer vectors
        """

        model = Vectors.create("transformers", "sentence-transformers/bert-base-nli-mean-tokens", False, None)

        # Generate enough volume to test batching
        documents = [(x, "This is a test", None) for x in range(1000)]

        ids, dimension, stream = model.index(documents)

        self.assertEqual(len(ids), 1000)
        self.assertEqual(dimension, 768)
        self.assertIsNotNone(os.path.exists(stream))
