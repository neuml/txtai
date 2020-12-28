"""
Vectors module tests
"""

import os
import tempfile
import unittest

from txtai.vectors import WordVectors, Vectors

class TestWordVectors(unittest.TestCase):
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

    def testIndex(self):
        """
        Test word vector indexing
        """

        # Generate data
        documents = [(x, "This is a test", None) for x in range(1000)]

        model = Vectors.create({"path": self.path}, None)

        ids, dimension, stream = model.index(documents)

        self.assertEqual(len(ids), 1000)
        self.assertEqual(dimension, 300)
        self.assertIsNotNone(os.path.exists(stream))

    def testTransform(self):
        """
        Test word vector transform
        """

        model = Vectors.create({"path": self.path}, None)
        self.assertEqual(len(model.transform((None, ["txtai"], None))), 300)

    def testNoExist(self):
        """
        Test loading model that doesn't exist
        """

        # Test non-existent path raises an exception
        with self.assertRaises(IOError):
            Vectors.create({"path": os.path.join(tempfile.gettempdir(), "noexist")}, None)

    def testLookup(self):
        """
        Test word vector lookup
        """

        model = Vectors.create({"path": self.path}, None)
        self.assertEqual(model.lookup(["txtai", "embeddings", "sentence"]).shape, (3, 300))

    def testBlocking(self):
        """
        Test blocking load of vector model
        """

        config = {"path": self.path}
        model = Vectors.create(config, None)

        self.assertFalse(model.initialized)

        config["ids"] = ["0", "1"]
        config["dimensions"] = 300
        model = Vectors.create(config, None)

        self.assertTrue(model.initialized)
