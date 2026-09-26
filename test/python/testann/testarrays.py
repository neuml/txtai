"""
Array-backed ANN deletion tests
"""

import os
import tempfile
import unittest

from itertools import product

import numpy as np

from txtai.ann import ANNFactory


class TestArrays(unittest.TestCase):
    """
    NumPy and Torch ANN tests.
    """

    def testDeleteBounds(self):
        """
        Test invalid row IDs cannot delete live vectors or raise indexing errors
        """

        for backend, quantize, ids in product(["numpy", "torch"], [None, 1], [[], [3, 99], [-1], [-3], [-4], [-1, 0, 0, 3, 99]]):
            with self.subTest(backend=backend, quantize=quantize, ids=ids):
                data = self.vectors(quantize)
                model = ANNFactory.create({"backend": backend, "dimensions": 2, "quantize": quantize})
                self.addCleanup(model.close)
                model.index(data.copy())
                model.delete(np.array(ids, dtype=np.int64))

                expected = data.copy()
                if 0 in ids:
                    expected[0] = 0
                np.testing.assert_array_equal(model.numpy(model.backend), expected)
                self.assertEqual(model.count(), 2 if 0 in ids else 3)
                self.assertEqual(model.search(data[2:], 1)[0][0][0], 2)

    def testDeleteSaveLoad(self):
        """
        Test mixed valid/invalid deletes remain correct after saving, loading and appending
        """

        for backend, quantize, safetensors in product(["numpy", "torch"], [None, 1], [False, True]):
            with self.subTest(backend=backend, quantize=quantize, safetensors=safetensors):
                data = self.vectors(quantize)
                config = {"backend": backend, "dimensions": 2, "quantize": quantize, backend: {"safetensors": safetensors}}
                model = ANNFactory.create(config)
                self.addCleanup(model.close)
                model.index(data.copy())
                model.delete([0, -1, 3])

                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "index")
                    model.save(path)
                    loaded = ANNFactory.create(dict(model.config))
                    self.addCleanup(loaded.close)
                    loaded.load(path)
                    self.assertEqual(loaded.count(), 2)
                    self.assertEqual(loaded.search(data[2:], 1)[0][0][0], 2)
                    loaded.append(data[:1].copy())
                    self.assertEqual(loaded.count(), 3)
                    self.assertEqual(loaded.search(data[:1], 1)[0][0][0], 3)

    def vectors(self, quantize):
        """
        Create normalized float vectors or packed binary vectors.
        """

        return np.array([[255, 0], [0, 255], [255, 255]], dtype=np.uint8) if quantize else np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
