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

    def testAnnoyCustom(self):
        """
        Test Annoy backend with custom settings
        """

        # Test with custom settings
        config = {"annoy": {"ntrees": 2, "searchk": 1}}

        self.assertEqual(self.backend("annoy", config).config["backend"], "annoy")
        self.assertIsNotNone(self.save("annoy", config))
        self.assertGreater(self.search("annoy", config), 0)

    @unittest.skipIf(os.name == "nt", "Faiss not installed on Windows")
    def testFaiss(self):
        """
        Test Faiss backend
        """

        self.assertEqual(self.backend("faiss").config["backend"], "faiss")
        self.assertEqual(self.backend("faiss", None, 5000).config["backend"], "faiss")

        self.assertIsNotNone(self.save("faiss"))
        self.assertGreater(self.search("faiss"), 0)

    @unittest.skipIf(os.name == "nt", "Faiss not installed on Windows")
    def testFaissCustom(self):
        """
        Test Faiss backend with custom settings
        """

        # Test with custom settings
        config = {"faiss": {"nprobe": 2, "components": "PCA16,IDMap,SQ8"}}

        self.assertEqual(self.backend("faiss", config).config["backend"], "faiss")
        self.assertIsNotNone(self.save("faiss", config))
        self.assertGreater(self.search("faiss", config), 0)

    def testHnsw(self):
        """
        Test Hnswlib backend
        """

        self.assertEqual(self.backend("hnsw").config["backend"], "hnsw")
        self.assertIsNotNone(self.save("hnsw"))
        self.assertGreater(self.search("hnsw"), 0)

    def testHnswCustom(self):
        """
        Test Hnswlib backend with custom settings
        """

        # Test with custom settings
        config = {"hnsw": {"efconstruction": 100, "m": 4, "randomseed": 0, "efsearch": 5}}

        self.assertEqual(self.backend("hnsw", config).config["backend"], "hnsw")
        self.assertIsNotNone(self.save("hnsw", config))
        self.assertGreater(self.search("hnsw", config), 0)

    def backend(self, name, params=None, length=10000):
        """
        Test a backend.

        Args:
            backend: backend name
            params: additional config parameters
            length: number of rows to generate

        Returns:
            ANN model
        """

        # Generate test data
        data = np.random.rand(length, 300).astype(np.float32)
        self.normalize(data)

        config = {"backend": name, "dimensions": data.shape[1]}
        if params:
            config.update(params)

        model = ANN.create(config)
        model.index(data)

        return model

    def save(self, backend, params=None):
        """
        Test save/load.

        Args:
            backend: backend name
            params: additional config parameters

        Returns:
            ANN model
        """

        model = self.backend(backend, params)

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "ann")

        model.save(index)
        model.load(index)

        return model

    def search(self, backend, params=None):
        """
        Test ANN search.

        Args:
            backend: backend name
            params: additional config parameters

        Returns:
            search results
        """

        # Generate ANN index
        model = self.backend(backend, params)

        # Generate query vector
        query = np.random.rand(300).astype(np.float32)
        self.normalize(query)

        # Ensure top result has similarity > 0
        return model.search(np.array([query]), 1)[0][0][1]

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
