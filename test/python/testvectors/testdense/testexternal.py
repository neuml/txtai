"""
External module tests
"""

import os
import tempfile
import unittest

from unittest.mock import patch

import numpy as np

from txtai import Embeddings
from txtai.vectors import External, VectorsFactory


class Transform:
    """
    Transform function
    """

    def __call__(self, data):
        return [[0.0, 1.0]]


class TestExternal(unittest.TestCase):
    """
    External vectors tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Create External vectors instance.
        """

        cls.model = VectorsFactory.create({"method": "external"}, None)

    def testDisabled(self):
        """
        Test that transforms are disabled by default
        """

        with self.assertRaises(ImportError):
            VectorsFactory.create({"transform": "testvectors.testdense.testexternal.Transform"}, None)

    def testIndex(self):
        """
        Test indexing with external vectors
        """

        # Generate dummy data
        data = np.random.rand(1000, 768).astype(np.float32)

        # Generate enough volume to test batching
        documents = [(x, data[x], None) for x in range(1000)]

        ids, dimension, batches, stream = self.model.index(documents)

        self.assertEqual(len(ids), 1000)
        self.assertEqual(dimension, 768)
        self.assertEqual(batches, 2)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (500, 768))

    def testIndexCheckpoint(self):
        """
        Test a changed precomputed vector is reindexed when resuming a checkpoint
        """

        original = np.zeros(1536, dtype=np.float32)
        original[759] = 1
        updated = np.zeros_like(original)
        updated[761] = 1

        with tempfile.TemporaryDirectory() as checkpoint:
            for vector in (original, updated):
                embeddings = Embeddings({"method": "external", "backend": "numpy"})
                embeddings.index([("doc", vector, None)], checkpoint=checkpoint)
                self.assertEqual(embeddings.search(vector, 1), [("doc", 1.0)])

    def testPrecomputed(self):
        """
        Test precomputed vectors are normalized the same way as encoded vectors
        """

        data = [np.array([3.0, 4.0], dtype=np.float32), np.array([20.0, 10.0], dtype=np.float32)]

        # Precomputed arrays skip encoding but must still be normalized
        embeddings = self.model.batchtransform([(None, x, None) for x in data])
        self.assertTrue(np.allclose(embeddings, [[0.6, 0.8], [0.894427, 0.447214]]))

        # Same output as the indexing path, which runs vectorize
        self.assertTrue(np.allclose(embeddings, self.model.vectorize(data, "data")))

    def testPrecomputedOptions(self):
        """
        Test precomputed vectors are truncated and quantized the same way as encoded vectors
        """

        data = [np.array([0.3, -0.4, 0.5, 0.1], dtype=np.float32), np.array([-1.0, 2.0, 0.5, 3.0], dtype=np.float32)]

        for config in [{"dimensionality": 2}, {"quantize": 8}, {"dimensionality": 2, "quantize": 1}]:
            model = VectorsFactory.create({"method": "external", **config}, None)
            embeddings = model.batchtransform([(None, x, None) for x in data])
            self.assertTrue(np.array_equal(embeddings, model.vectorize(data, "data")), config)

    def testQuantizeSimilarity(self):
        """
        Test similarity with quantized vectors scores the same as search
        """

        vectors = {str(x): row for x, row in enumerate(np.random.rand(10, 50) - 0.5)}
        texts = list(vectors)

        for bits in [1, 3, 8]:
            embeddings = Embeddings({"transform": lambda data: [vectors[x] for x in data], "quantize": bits})
            embeddings.index(texts)

            results = dict(embeddings.search("0", len(texts)))
            for uid, score in embeddings.similarity("0", texts):
                self.assertAlmostEqual(score, results[uid], places=5)

    @patch.dict(os.environ, {"ALLOW_RESOLVE_TRANSFORM": "True"})
    def testResolution(self):
        """
        Test resolving an external transform function
        """

        transform = VectorsFactory.create({"transform": "testvectors.testdense.testexternal.Transform"}, None)
        self.assertTrue(np.array_equal(transform.encode(["test"]), np.array([[0.0, 1.0]])))

    def testMethod(self):
        """
        Test method is derived when transform function passed
        """

        model = VectorsFactory.create({"transform": lambda _: [[0.0, 1.0]]}, None)
        self.assertTrue(isinstance(model, External))
