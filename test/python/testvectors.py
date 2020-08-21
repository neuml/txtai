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

        # Build word vectors on README file
        WordVectors.build("README.md", 300, 3, path)

        # Save model path
        self.path = path + ".magnitude"

    def testLoad(self):
        model = Vectors.create("words", self.path, True, None)
        self.assertEqual(len(model.transform((None, ["txtai"], None))), 300)

    def testLookup(self):
        model = Vectors.create("words", self.path, True, None)
        self.assertEqual(model.lookup(["txtai", "embeddings", "sentence"]).shape, (3, 300))
