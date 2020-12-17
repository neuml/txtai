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

    def setUp(self):
        """
        Test a WordVectors build.
        """

        # Word vectors path
        path = os.path.join(tempfile.gettempdir(), "vectors")

        # Save model path
        self.path = path + ".magnitude"

        # Build word vectors, if they don't already exist
        if not os.path.exists(self.path):
            WordVectors.build("README.md", 300, 3, path)

    def testLoad(self):
        """
        Test loading word vectors
        """

        model = Vectors.create("words", self.path, True, None)
        self.assertEqual(len(model.transform((None, ["txtai"], None))), 300)

    def testLookup(self):
        """
        Test word vector lookup
        """

        model = Vectors.create("words", self.path, True, None)
        self.assertEqual(model.lookup(["txtai", "embeddings", "sentence"]).shape, (3, 300))
