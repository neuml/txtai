"""
GGML ANN module tests
"""

import os
import tempfile

import ggml
import numpy as np

from txtai.ann import ANNFactory
from txtai.ann.dense.ggml import GGMLTensors

from .base import DenseTest


class TestGGML(DenseTest):
    """
    GGML ANN tests.
    """

    def testGGML(self):
        """
        Test GGML backend
        """

        self.runTests("ggml")

    def testGGMLDelete(self):
        """
        Test GGML backend deletes
        """

        ann = ANNFactory.create({"backend": "ggml"})

        # Generate and index dummy data
        data = np.random.rand(100, 256).astype(np.float32)
        ann.index(data)

        # Validate count
        self.assertEqual(ann.count(), 100)

        # Test delete
        ann.delete([0])
        self.assertEqual(ann.count(), 99)

        # Deleting the same id again is a no-op for the count
        ann.delete([0])
        self.assertEqual(ann.count(), 99)

        # Save updated index with deletes and reload
        index = os.path.join(tempfile.gettempdir(), "ggml.deletes")
        ann.save(index)
        ann.load(index)
        self.assertEqual(ann.count(), 99)

    def testGGMLTopn(self):
        """
        Test that GGML searches return the same top scores as a full sort for all limit sizes
        """

        rng = np.random.default_rng(0)
        data = rng.random((50, 8), dtype=np.float32)
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        queries = data[:3] + 0.01

        ann = ANNFactory.create({"backend": "ggml", "ggml": {"gpu": False}})
        ann.index(data.copy())

        scores = np.dot(queries, data.T)

        # Limits that are zero, smaller than, equal to and larger than the number of rows
        for limit in (0, 1, 5, 49, 50, 80):
            with self.subTest(limit=limit):
                results = ann.search(queries, limit)

                for x, result in enumerate(results):
                    expected = np.sort(scores[x])[::-1][:limit]

                    self.assertEqual(len(result), len(expected))
                    np.testing.assert_allclose([score for _, score in result], expected, rtol=1e-5)

                    # Ids are unique and each id maps to its own score
                    ids = [uid for uid, _ in result]
                    self.assertEqual(len(set(ids)), len(ids))
                    np.testing.assert_allclose([scores[x][uid] for uid in ids], [score for _, score in result], rtol=1e-5)

        # Deleted rows are not returned
        best = ann.search(queries[:1], 1)[0][0][0]
        ann.delete([best])
        self.assertNotIn(best, [uid for uid, _ in ann.search(queries[:1], 5)[0]])

    def testGGMLQuantizeDisabled(self):
        """
        Test that quantize: false disables GGML tensor quantization
        """

        data = np.random.rand(4, 256).astype(np.float32)
        for quantize, expected in [(False, ggml.GGML_TYPE_F32), (True, ggml.GGML_TYPE_Q8_0), (4, ggml.GGML_TYPE_Q4_0)]:
            with self.subTest(quantize=quantize):
                tensors = GGMLTensors(False, 64, quantize)
                self.assertEqual(tensors.tensortype(data), expected)

    def testGGMLEmpty(self):
        """
        Test GGML backend with an empty index
        """

        ann = ANNFactory.create({"backend": "ggml", "dimensions": 240})

        # Index an empty array
        ann.index(np.zeros((0, 240), dtype=np.float32))
        self.assertEqual(ann.count(), 0)

        # Test save and load
        index = os.path.join(tempfile.gettempdir(), "ggml.empty")
        ann.save(index)
        ann.load(index)
        self.assertEqual(ann.count(), 0)

        # Append data to the loaded empty index
        data = np.random.rand(10, 240).astype(np.float32)
        self.normalize(data)
        ann.append(data)
        self.assertEqual(ann.count(), 10)

        # Generate query vector and test search
        query = np.random.rand(240).astype(np.float32)
        self.normalize(query)
        self.assertGreater(ann.search(np.array([query]), 1)[0][0][1], 0)

    def testGGMLQuantization(self):
        """
        Test GGML backend with quantization enabled
        """

        ann = ANNFactory.create({"backend": "ggml", "ggml": {"quantize": "Q4_0"}})

        # Generate and index dummy data
        data = np.random.rand(100, 256).astype(np.float32)
        ann.index(data)

        # Test save and load
        index = os.path.join(tempfile.gettempdir(), "ggml.q4_0.v1")
        ann.save(index)
        ann.load(index)

        # Generate query vector and test search
        query = np.random.rand(256).astype(np.float32)
        self.normalize(query)
        self.assertGreater(ann.search(np.array([query]), 1)[0][0][1], 0)

        # Validate count
        self.assertEqual(ann.count(), 100)

        # Test delete
        ann.delete([0])
        self.assertEqual(ann.count(), 99)

        # Save updated index with deletes and reload
        index = os.path.join(tempfile.gettempdir(), "ggml.q4_0.v2")
        ann.save(index)
        ann.load(index)
        ann.index(data)

    def testGGMLInvalid(self):
        """
        Test invalid GGML configurations
        """

        data = np.random.rand(100, 240).astype(np.float32)

        with self.assertRaises(ValueError):
            ann = ANNFactory.create({"backend": "ggml", "ggml": {"quantize": "NOEXIST", "gpu": False}})
            ann.index(data)

        with self.assertRaises(ValueError):
            ann = ANNFactory.create({"backend": "ggml", "ggml": {"quantize": "Q4_K"}})
            ann.index(data)

    def testGGMLSingleRow(self):
        """
        Test GGML backend with a single row index
        """

        ann = ANNFactory.create({"backend": "ggml", "dimensions": 240})

        # Generate and index a single row
        data = np.random.rand(1, 240).astype(np.float32)
        self.normalize(data)
        ann.index(data)
        self.assertEqual(ann.count(), 1)

        # Test save and load
        index = os.path.join(tempfile.gettempdir(), "ggml.single")
        ann.save(index)
        ann.load(index)
        self.assertEqual(ann.count(), 1)
        self.assertEqual(ann.search(data, 1)[0][0][0], 0)

        # Append a row
        ann.append(data)
        self.assertEqual(ann.count(), 2)

        # Test delete, including an out of range id
        ann.delete([0, 5])
        self.assertEqual(ann.count(), 1)
