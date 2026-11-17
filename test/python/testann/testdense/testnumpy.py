"""
NumPy ANN module tests
"""

import os
import tempfile

from unittest.mock import patch

import numpy as np

from txtai.ann import ANNFactory
from txtai.serialize import SerializeFactory

from .base import DenseTest


class TestNumPy(DenseTest):
    """
    NumPy ANN tests.
    """

    def testArrayDeleteBounds(self):
        """
        Invalid array deletion IDs must not wrap to live rows or raise IndexError.
        """

        for backend in ("numpy", "torch"):
            for quantize in (None, 1):
                with self.subTest(backend=backend, quantize=quantize):
                    data = np.array([[255, 0], [0, 255], [255, 255]], dtype=np.uint8) if quantize else np.eye(3, dtype=np.float32)
                    ann = ANNFactory.create({"backend": backend, "dimensions": data.shape[1], "quantize": quantize})
                    self.addCleanup(ann.close)
                    ann.index(data.copy())
                    ann.delete([-1, -4, 0, 0, 3, 99])
                    expected = data.copy()
                    expected[0] = 0
                    np.testing.assert_array_equal(ann.numpy(ann.backend), expected)
                    self.assertEqual(ann.count(), 2)

    def testTopn(self):
        """
        Test that searches return the same top scores as a full sort for all limit sizes
        """

        rng = np.random.default_rng(0)
        data = rng.random((50, 8), dtype=np.float32)
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        queries = data[:3] + 0.01

        for backend in ("numpy", "torch"):
            ann = ANNFactory.create({"backend": backend, "dimensions": data.shape[1]})
            self.addCleanup(ann.close)
            ann.index(data.copy())

            scores = np.dot(queries, data.T)

            # Limits that are zero, smaller than, equal to and larger than the number of rows
            for limit in (0, 1, 5, 49, 50, 80):
                with self.subTest(backend=backend, limit=limit):
                    results = ann.search(queries, limit)

                    for x, result in enumerate(results):
                        expected = np.sort(scores[x])[::-1][:limit]

                        self.assertEqual(len(result), len(expected))
                        np.testing.assert_allclose([score for _, score in result], expected, rtol=1e-5)

                        # Ids are unique and each id maps to its own score
                        ids = [uid for uid, _ in result]
                        self.assertEqual(len(set(ids)), len(ids))
                        np.testing.assert_allclose([scores[x][uid] for uid in ids], [score for _, score in result], rtol=1e-5)

    def testNumPy(self):
        """
        Test NumPy backend
        """

        self.runTests("numpy")

    @patch.dict(os.environ, {"ALLOW_PICKLE": "True"})
    def testNumPyLegacy(self):
        """
        Test NumPy backend with legacy pickled data
        """

        serializer = SerializeFactory.create("pickle", allowpickle=True)

        # Create output directory
        output = os.path.join(tempfile.gettempdir(), "ann.npy")
        path = os.path.join(output, "embeddings")
        os.makedirs(output, exist_ok=True)

        # Generate data and save as pickle
        data = np.random.rand(100, 240).astype(np.float32)
        serializer.save(data, path)

        ann = ANNFactory.create({"backend": "numpy"})
        ann.load(path)

        # Validate count
        self.assertEqual(ann.count(), 100)

    def testNumPySafetensors(self):
        """
        Test NumPy backend with safetensors storage
        """

        ann = ANNFactory.create({"backend": "numpy", "numpy": {"safetensors": True}})

        # Generate and index dummy data
        data = np.random.rand(100, 240).astype(np.float32)
        ann.index(data)

        # Test save and load
        index = os.path.join(tempfile.gettempdir(), "numpy.safetensors")
        ann.save(index)
        ann.load(index)

        # Generate query vector and test search
        query = np.random.rand(240).astype(np.float32)
        self.normalize(query)
        self.assertGreater(ann.search(np.array([query]), 1)[0][0][1], 0)

        # Validate count
        self.assertEqual(ann.count(), 100)

    def testNumPyQuantizeDelete(self):
        """
        Test that deleted rows don't resurface in search results for quantized (hamming) NumPy indexes
        """

        ann = ANNFactory.create({"backend": "numpy", "quantize": 1, "dimensions": 4})

        # Index uint8 vectors
        data = np.array([[1, 0, 0, 0], [0, 255, 0, 0], [0, 0, 255, 0], [0, 0, 0, 255]], dtype=np.uint8)
        ann.index(data)

        # Delete the first row and search with its (former) vector
        ann.delete([0])
        results = ann.search(np.array([data[0]]), 4)[0]

        # Deleted row must score 0 so the caller's score > 0 filter drops it, same as the dot-product path
        self.assertEqual(dict(results).get(0), 0)
        self.assertEqual(ann.count(), 3)
