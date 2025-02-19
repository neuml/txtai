"""
Vectors module tests
"""

import os
import tempfile
import unittest

import numpy as np

from txtai.vectors import Vectors, Recovery


class TestVectors(unittest.TestCase):
    """
    Vectors tests.
    """

    def testNotImplemented(self):
        """
        Test exceptions for non-implemented methods
        """

        vectors = Vectors(None, None, None)

        self.assertRaises(NotImplementedError, vectors.load, None)
        self.assertRaises(NotImplementedError, vectors.encode, None)

    def testNormalize(self):
        """
        Test batch normalize and single input normalize are equal
        """

        vectors = Vectors(None, None, None)

        # Generate data
        data1 = np.random.rand(5, 5).astype(np.float32)
        data2 = data1.copy()

        # Keep original data to ensure it changed
        original = data1.copy()

        # Normalize data
        vectors.normalize(data1)
        for x in data2:
            vectors.normalize(x)

        # Test both data arrays are the same and changed from original
        self.assertTrue(np.allclose(data1, data2))
        self.assertFalse(np.allclose(data1, original))

    def testRecovery(self):
        """
        Test vectors recovery failure
        """

        # Checkpoint directory
        checkpoint = os.path.join(tempfile.gettempdir(), "recovery")
        os.makedirs(checkpoint, exist_ok=True)

        # Create empty file
        # pylint: disable=R1732
        f = open(os.path.join(checkpoint, "id"), "w", encoding="utf-8")
        f.close()

        # Create the recovery instance with an empty checkpoint file
        recovery = Recovery(checkpoint, "id")
        self.assertIsNone(recovery())
