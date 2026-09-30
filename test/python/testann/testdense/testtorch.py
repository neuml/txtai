"""
Torch ANN module tests
"""

import os
import platform
import tempfile
import unittest

import numpy as np

from txtai.ann import ANNFactory

from .base import DenseTest


class TestTorch(DenseTest):
    """
    Torch ANN tests.
    """

    def testTorch(self):
        """
        Test Torch backend
        """

        self.runTests("torch")

    def testTorchDelete(self):
        """
        Test Torch backend delete keeps the stored data type for quantized (uint8) and float16 indexes
        """

        for dtype, config in [(np.uint8, {"quantize": 1}), (np.float16, {})]:
            with self.subTest(dtype=dtype.__name__):
                ann = ANNFactory.create({"backend": "torch", "dimensions": 4, **config})

                # Index vectors with a single non-zero value per row
                data = np.array([[1, 0, 0, 0], [0, 255, 0, 0], [0, 0, 255, 0], [0, 0, 0, 255]], dtype=dtype)
                ann.index(data)

                # Delete the first row and search with its (former) vector
                ann.delete([0])
                results = ann.search(np.array([data[0]]), 4)[0]

                # Deleted row must score 0 so the caller's score > 0 filter drops it and the data type must be unchanged
                self.assertEqual(dict(results).get(0), 0)
                self.assertEqual(ann.count(), 3)
                self.assertEqual(str(ann.backend.dtype), f"torch.{dtype.__name__}")

    @unittest.skipIf(platform.system() == "Darwin", "Torch quantization not supported on macOS")
    def testTorchQuantization(self):
        """
        Test Torch backend with quantization enabled
        """

        for qtype in ["fp4", "nf4", "int8"]:
            ann = ANNFactory.create({"backend": "torch", "torch": {"quantize": {"type": qtype}}})

            # Generate and index dummy data
            data = np.random.rand(100, 240).astype(np.float32)
            ann.index(data)

            # Test save and load
            index = os.path.join(tempfile.gettempdir(), f"{qtype}.safetensors")
            ann.save(index)
            ann.load(index)

            # Generate query vector and test search
            query = np.random.rand(240).astype(np.float32)
            self.normalize(query)
            self.assertGreater(ann.search(np.array([query]), 1)[0][0][1], 0)

            # Validate count
            self.assertEqual(ann.count(), 100)

            # Test delete
            ann.delete([0])
