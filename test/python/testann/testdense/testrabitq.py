"""
RaBitQ ANN module tests
"""

import os
import tempfile
import time

import numpy as np

from rabitqlib import HnswIndex, IvfIndex

from txtai.ann import ANNFactory

from .base import DenseTest


class TestRaBitQ(DenseTest):
    """
    RaBitQ ANN tests.
    """

    def testRabitQ(self):
        """
        Test RaBitQ backend
        """

        self.runTests("rabitq", None, False)

    def testRabitQCustom(self):
        """
        Test RaBitQ backend with custom settings
        """

        # Test with custom settings
        self.runTests("rabitq", {"rabitq": {"mode": "hnsw"}}, False)
        self.runTests("rabitq", {"rabitq": {"clusters": 8, "nprobe": 2}}, False)
        self.runTests("rabitq", {"rabitq": {"nbits": 4}}, False)
        self.runTests("rabitq", {"rabitq": {"mode": "hnsw", "nbits": 4}}, False)
        self.runTests("rabitq", {"rabitq": {"nbits": 32}}, False)

        # Generate dummy data
        data = np.random.rand(100, 240).astype(np.float32)
        self.normalize(data)

        # Test invalid modes and quantization bits
        for mode, nbits in [("invalid", 1), ("ivf", 10), ("hnsw", 32)]:
            with self.assertRaises(ValueError):
                ann = ANNFactory.create({"backend": "rabitq", "dimensions": 240, "rabitq": {"mode": mode, "nbits": nbits}})
                ann.index(data)

        # Test a failed load leaves an empty index
        ann = ANNFactory.create({"backend": "rabitq", "dimensions": 240})
        ann.index(data)
        index = os.path.join(tempfile.gettempdir(), f"rabitq.invalid.{round(time.time() * 1000)}")
        ann.save(index)
        ann.config["rabitq"] = {"mode": "invalid"}
        with self.assertRaises(ValueError):
            ann.load(index)
        self.assertEqual(ann.count(), 0)

    def testRabitQUpdate(self):
        """
        Test RabitQ stores a single file and does not support append and delete
        """

        # Generate dummy data
        data = np.random.rand(100, 240).astype(np.float32)
        self.normalize(data)

        for mode in ["ivf", "hnsw"]:
            ann = ANNFactory.create({"backend": "rabitq", "dimensions": 240, "rabitq": {"mode": mode, "nprobe": 100}})
            ann.index(data)

            # Offset marks the index as existing, see Embeddings.exists
            self.assertEqual(ann.config["offset"], 100)

            with self.assertRaises(NotImplementedError):
                ann.append(data[:1])

            with self.assertRaises(NotImplementedError):
                ann.delete([0])

            # Limit above the row count returns every row
            for result in ann.search(data[:2], 200):
                self.assertEqual(len(result), 100)

            # Index is a single file that reloads with the same count and results
            index = os.path.join(tempfile.gettempdir(), f"rabitq.{mode}.{round(time.time() * 1000)}")
            ann.save(index)
            self.assertTrue(os.path.isfile(index))

            # File is the native index, readable without the wrapper
            native = IvfIndex.load(index) if mode == "ivf" else HnswIndex.load(index)
            self.assertEqual(native.max_elements, 100)

            expected = ann.search(data[:2], 10)
            ann.load(index)
            self.assertEqual(ann.count(), 100)
            self.assertEqual(ann.search(data[:2], 10), expected)
