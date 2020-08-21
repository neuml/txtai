"""
ANN module tests
"""

import os
import tempfile
import unittest

import numpy as np

from txtai.ann import ANN

class TestANN(unittest.TestCase):
    """
    ANN tests
    """

    def testAnnoy(self):
        """
        Test Annoy backend
        """

        self.assertEqual(self.backend("annoy").config["backend"], "annoy")

    @unittest.skipIf(os.name == "nt", "Faiss not installed on Windows")
    def testFaiss(self):
        """
        Test Faiss backend
        """

        self.assertEqual(self.backend("faiss").config["backend"], "faiss")
        self.assertEqual(self.backend("faiss", 5000).config["backend"], "faiss")

    def testHnsw(self):
        """
        Test Hnswlib
        """

        self.assertEqual(self.backend("hnsw").config["backend"], "hnsw")

    def testSave(self):
        """
        Tests ANN save/load.
        """

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "ann")

        model = self.backend("annoy")
        model.save(index)
        model.load(index)

    def backend(self, name, length=100):
        """
        Tests a backend
        """

        # Generate test data
        data = np.random.rand(length, 300).astype(np.float32)

        model = ANN.create({"backend": name, "dimensions": data.shape[1]})
        model.index(data)

        return model
