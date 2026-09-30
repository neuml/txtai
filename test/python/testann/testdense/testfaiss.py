"""
Faiss ANN module tests
"""

import os
import sys
import unittest

from unittest.mock import patch

import numpy as np

from txtai.ann import ANNFactory

from .base import DenseTest


class TestFaiss(DenseTest):
    """
    Faiss ANN tests.
    """

    def testFaiss(self):
        """
        Test Faiss backend
        """

        self.runTests("faiss")

    def testFaissBinary(self):
        """
        Test Faiss backend with a binary hash index
        """

        ann = ANNFactory.create({"backend": "faiss", "quantize": 1, "dimensions": 240 * 8, "faiss": {"components": "BHash32"}})

        # Generate and index dummy data
        data = np.random.rand(100, 240).astype(np.uint8)
        ann.index(data)

        # Generate query vector and test search
        query = np.random.rand(240).astype(np.uint8)
        self.assertGreater(ann.search(np.array([query]), 1)[0][0][1], 0)

    def testFaissCustom(self):
        """
        Test Faiss backend with custom settings
        """

        # Test with custom settings
        self.runTests("faiss", {"faiss": {"nprobe": 2, "components": "PCA16,IDMap,SQ8", "sample": 1.0}}, False)
        self.runTests("faiss", {"faiss": {"components": "IVF,SQ8"}}, False)

    @patch("platform.system")
    def testFaissMacOS(self, system):
        """
        Test Faiss backend with macOS
        """

        # Run test
        system.return_value = "Darwin"

        # pylint: disable=C0415, W0611
        # Force reload of class
        name = "txtai.ann.dense.faiss"
        module = sys.modules[name]
        del sys.modules[name]
        import txtai.ann.dense.faiss

        # Run tests
        self.runTests("faiss")

        # Restore original module
        sys.modules[name] = module

    @unittest.skipIf(os.name == "nt", "mmap not supported on Windows")
    def testFaissMmap(self):
        """
        Test Faiss backend with mmap enabled
        """

        # Test to with mmap enabled
        self.runTests("faiss", {"faiss": {"mmap": True}}, False)

    def testFaissNprobeClamp(self):
        """
        Test Faiss default nprobe clamp
        """

        ann, _ = self.faissmodel(5001, {"sample": 0.01})
        self.assertGreater(round(ann.cells(ann.count()) / 16), ann.backend.nlist)
        self.assertEqual(ann.nprobe(), ann.backend.nlist)

    def testFaissNprobeConfigured(self):
        """
        Test configured Faiss nprobe values
        """

        for nprobe in [1, 3]:
            with self.subTest(nprobe=nprobe):
                ann, _ = self.faissmodel(100, {"components": "IVF2,Flat", "nprobe": nprobe})
                self.assertEqual(ann.nprobe(), nprobe)

    def testFaissNprobeResults(self):
        """
        Test Faiss default nprobe clamp results
        """

        ann, rng = self.faissmodel(5001, {"sample": 0.01})
        queries = rng.random((10, 8), dtype=np.float32)
        self.normalize(queries)

        default = ann.search(queries, 5)
        ann.config["faiss"]["nprobe"] = ann.backend.nlist
        full = ann.search(queries, 5)

        self.assertEqual(default, full)

    def testFaissNprobeSmall(self):
        """
        Test Faiss default nprobe clamp on a small index
        """

        ann, _ = self.faissmodel(100, {"components": "IVF2,Flat"})
        self.assertEqual(ann.backend.nlist, 2)
        self.assertEqual(ann.nprobe(), ann.backend.nlist)

    def testFaissNprobeUnchanged(self):
        """
        Test Faiss default nprobe without a clamp
        """

        ann, _ = self.faissmodel(5001, {"components": "IVF16,Flat"})
        expected = round(ann.cells(ann.count()) / 16)
        self.assertLessEqual(expected, ann.backend.nlist)
        self.assertEqual(ann.nprobe(), expected)

    def testFaissQuantizeDisabled(self):
        """
        Test that quantize: false disables Faiss storage quantization
        """

        for quantize, expected in [(False, "IDMap,Flat"), (True, "IDMap,SQ8")]:
            with self.subTest(quantize=quantize):
                ann = ANNFactory.create({"backend": "faiss", "dimensions": 4, "quantize": quantize})
                self.assertEqual(ann.configure(100, 100), expected)
