"""
External module tests
"""

import os
import unittest

from unittest.mock import patch

import numpy as np

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
