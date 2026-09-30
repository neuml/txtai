"""
Base class for ANN Dense tests
"""

import os
import tempfile
import time
import unittest

import numpy as np

from txtai.ann import ANNFactory


class DenseTest(unittest.TestCase):
    """
    Base class for ANN Dense tests.
    """

    def runTests(self, name, params=None, update=True):
        """
        Runs a series of standard backend tests.

        Args:
            name: backend name
            params: additional config parameters
            update: If append/delete options should be tested
        """

        self.assertEqual(self.backend(name, params).config["backend"], name)
        self.assertEqual(self.save(name, params).count(), 10000)

        if update:
            self.assertEqual(self.append(name, params, 500).count(), 10500)
            self.assertEqual(self.delete(name, params, [0, 1]).count(), 9998)
            self.assertEqual(self.delete(name, params, [100000]).count(), 10000)

        self.assertGreater(self.search(name, params), 0)
        self.assertGreater(self.limit(name, params), 0)

    def backend(self, name, params=None, length=10000):
        """
        Test a backend.

        Args:
            name: backend name
            params: additional config parameters
            length: number of rows to generate

        Returns:
            ANN model
        """

        # Generate test data
        data = np.random.rand(length, 240).astype(np.float32)
        self.normalize(data)

        config = {"backend": name, "dimensions": data.shape[1]}
        if params:
            config.update(params)

        model = ANNFactory.create(config)
        model.index(data)

        return model

    def faissmodel(self, length, settings):
        """
        Builds a seeded Faiss model.

        Args:
            length: number of rows to generate
            settings: Faiss settings

        Returns:
            Faiss model and random number generator
        """

        rng = np.random.default_rng(0)
        data = rng.random((length, 8), dtype=np.float32)
        self.normalize(data)

        model = ANNFactory.create({"backend": "faiss", "dimensions": data.shape[1], "faiss": settings})
        model.index(data)

        return model, rng

    def append(self, name, params=None, length=500):
        """
        Appends new data to index.

        Args:
            name: backend name
            params: additional config parameters
            length: number of rows to generate

        Returns:
            ANN model
        """

        # Initial model
        model = self.backend(name, params)

        # Generate test data
        data = np.random.rand(length, 240).astype(np.float32)
        self.normalize(data)

        model.append(data)

        return model

    def delete(self, name, params=None, ids=None):
        """
        Deletes data from index.

        Args:
            name: backend name
            params: additional config parameters
            ids: ids to delete

        Returns:
            ANN model
        """

        # Initial model
        model = self.backend(name, params)
        model.delete(ids)

        return model

    def deletenumpy(self, name, params=None):
        """
        Test deleting ids passed as a NumPy array. Only the requested rows should be deleted.

        Args:
            name: backend name
            params: additional config parameters
        """

        model = self.backend(name, params, 10)
        model.delete(np.array([3, 4]))

        # Generate query vector
        query = np.random.rand(240).astype(np.float32)
        self.normalize(query)

        self.assertEqual(model.count(), 8)
        self.assertEqual(sorted(uid for uid, _ in model.search(np.array([query]), 10)[0]), [0, 1, 2, 5, 6, 7, 8, 9])

    def save(self, name, params=None):
        """
        Test save/load.

        Args:
            name: backend name
            params: additional config parameters

        Returns:
            ANN model
        """

        model = self.backend(name, params)

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), f"ann.{name}.{round(time.time() * 1000)}")

        # Generate query vector
        query = np.random.rand(240).astype(np.float32)
        self.normalize(query)

        # Save index and ensure it's still searchable
        model.save(index)
        self.assertGreater(model.search(np.array([query]), 1)[0][0][1], 0)

        # Close and reload index
        model.close()
        model.load(index)

        # Ensure reloaded index is searchable
        self.assertGreater(model.search(np.array([query]), 1)[0][0][1], 0)

        return model

    def search(self, name, params=None):
        """
        Test ANN search.

        Args:
            name: backend name
            params: additional config parameters

        Returns:
            search results
        """

        # Generate ANN index
        model = self.backend(name, params)

        # Generate query vector
        query = np.random.rand(240).astype(np.float32)
        self.normalize(query)

        # Ensure top result has similarity > 0
        return model.search(np.array([query]), 1)[0][0][1]

    def limit(self, name, params=None):
        """
        Test ANN limit search.

        Args:
            name: backend name
            params: additional config parameters

        Returns:
            search results
        """

        # Generate ANN index
        model = self.backend(name, params, 50)

        # Generate query vector
        query = np.random.rand(240).astype(np.float32)
        self.normalize(query)

        # Ensure limit being > count doesn't throw an error
        return model.search(np.array([query]), 100)[0][0][1]

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
