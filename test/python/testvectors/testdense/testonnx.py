"""
ONNX module tests
"""

import os
import tempfile
import unittest

import numpy as np

from txtai.pipeline import HFOnnx
from txtai.vectors import VectorsFactory


class TestONNX(unittest.TestCase):
    """
    ONNX vectors tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Export a model to ONNX and create an ONNX vectors instance.
        """

        path = "sentence-transformers/paraphrase-MiniLM-L3-v2"

        # Export model to ONNX
        cls.path = os.path.join(tempfile.gettempdir(), "vectors", "model.onnx")
        os.makedirs(os.path.dirname(cls.path), exist_ok=True)
        HFOnnx()(path, "pooling", cls.path, True)

        cls.model = VectorsFactory.create({"path": cls.path, "tokenizer": path, "gpu": False}, None)

    def testMethod(self):
        """
        Test that an .onnx path resolves to the onnx method
        """

        self.assertEqual(VectorsFactory.method({"path": self.path}), "onnx")
        self.assertEqual(VectorsFactory.method({"path": "model.tflite"}), "litert")

    def testIndex(self):
        """
        Test indexing with ONNX vectors
        """

        ids, dimension, batches, stream = self.model.index([(0, "test", None)])

        self.assertEqual(len(ids), 1)
        self.assertEqual(dimension, 384)
        self.assertEqual(batches, 1)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (1, 384))

    def testEncodeBatch(self):
        """
        Test that results are stable when a batch spans multiple encode batches
        """

        data = ["dog", "puppy", "quantum chromodynamics", "cat", "kitten"]

        single = self.model.encode(data)

        self.model.encodebatch = 2
        batched = self.model.encode(data)
        self.model.encodebatch = 32

        self.assertEqual(single.shape, (5, 384))
        self.assertTrue(np.allclose(single, batched, atol=1e-5))

    def testSimilarity(self):
        """
        Test that pooled embeddings carry semantics
        """

        embeddings = self.model.encode(["dog", "puppy", "quantum chromodynamics"])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.assertGreater(float(embeddings[0] @ embeddings[1]), float(embeddings[0] @ embeddings[2]))
