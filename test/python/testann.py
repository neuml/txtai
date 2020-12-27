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
        self.assertIsNotNone(self.save("annoy"))
        self.assertGreater(self.search("annoy"), 0)

    @unittest.skipIf(os.name == "nt", "Faiss not installed on Windows")
    def testFaiss(self):
        """
        Test Faiss backend
        """

        self.assertEqual(self.backend("faiss").config["backend"], "faiss")
        self.assertEqual(self.backend("faiss", 5000).config["backend"], "faiss")

        self.assertIsNotNone(self.save("faiss"))
        self.assertGreater(self.search("faiss"), 0)

    def testHnsw(self):
        """
        Test Hnswlib backend
        """

        self.assertEqual(self.backend("hnsw").config["backend"], "hnsw")
        self.assertIsNotNone(self.save("hnsw"))
        self.assertGreater(self.search("hnsw"), 0)

    def backend(self, name, length=100):
        """
        Test a backend
        """

        # Generate test data
        data = np.random.rand(length, 300).astype(np.float32)
        self.normalize(data)

        model = ANN.create({"backend": name, "dimensions": data.shape[1]})
        model.index(data)

        return model

    def save(self, backend):
        """
        Test save/load
        """

        model = self.backend(backend)

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "ann")

        model.save(index)
        model.load(index)

        return model

    def search(self, backend):
        """
        Test ANN search
        """

        # Generate ANN index
        model = self.backend(backend)

        # Generate query vector
        query = np.random.rand(300).astype(np.float32)
        self.normalize(query)

        # Ensure top result has similarity > 0
        return model.search(query, 1)[0][1]

    def normalize(self, embeddings):
        """
        Normalizes embeddings using L2 normalization. Operation applied directly on array.

        Args:
            embeddings: input embeddings matrix
        """

        # Calculation is different for matrices vs vectors
        if len(embeddings.shape) > 1:
            embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        else:
            embeddings /= np.linalg.norm(embeddings)
