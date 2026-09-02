"""
ONNX module tests
"""

import os
import shutil
import tempfile
import unittest

from unittest.mock import patch

import numpy as np
import onnx

from onnx import helper, TensorProto
from tokenizers import Tokenizer, models, pre_tokenizers

from txtai.pipeline import HFOnnx
from txtai.vectors import VectorsFactory
from txtai.vectors.dense.onnx import ONNX


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
        HFOnnx()(path, "pooling", cls.path)

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

        # Each input sees a different amount of padding depending on how the data is batched,
        # so compare direction rather than exact values
        single = single / np.linalg.norm(single, axis=1, keepdims=True)
        batched = batched / np.linalg.norm(batched, axis=1, keepdims=True)
        self.assertTrue(np.all(np.sum(single * batched, axis=1) > 0.99))

    def testSimilarity(self):
        """
        Test that pooled embeddings carry semantics
        """

        embeddings = self.model.encode(["dog", "puppy", "quantum chromodynamics"])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.assertGreater(float(embeddings[0] @ embeddings[1]), float(embeddings[0] @ embeddings[2]))


class TestONNXModels(unittest.TestCase):
    """
    ONNX vectors tests covering model shapes and configuration a single exported model doesn't reach.
    These build small ONNX graphs locally, no model download required.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create a working directory and a small tokenizer.
        """

        cls.directory = os.path.join(tempfile.gettempdir(), "onnxvectors")
        os.makedirs(cls.directory, exist_ok=True)

        vocab = {"[PAD]": 0, "[UNK]": 1, "dog": 2, "puppy": 3, "quantum": 4, "physics": 5}
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        cls.tokenizer = os.path.join(cls.directory, "tokenizer.json")
        tokenizer.save(cls.tokenizer)

        cls.vocab = len(vocab)

    def build(self, name, pooled=False, types=False):
        """
        Builds a small ONNX model that maps input ids to hidden states.

        Args:
            name: model file name
            pooled: True to output a single vector per input, False for token level output
            types: True to declare a token_type_ids input

        Returns:
            model path
        """

        table = np.eye(self.vocab, 4, dtype=np.float32)
        initializers = [helper.make_tensor("table", TensorProto.FLOAT, table.shape, table.flatten())]

        inputs = [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "sequence"]),
        ]
        if types:
            inputs.append(helper.make_tensor_value_info("token_type_ids", TensorProto.INT64, ["batch", "sequence"]))

        nodes = [helper.make_node("Gather", ["table", "input_ids"], ["hidden"], axis=0)]
        if pooled:
            nodes.append(helper.make_node("ReduceMean", ["hidden"], ["output"], axes=[1], keepdims=0))
            outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 4])]
        else:
            nodes.append(helper.make_node("Identity", ["hidden"], ["output"]))
            outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", "sequence", 4])]

        graph = helper.make_graph(nodes, "model", inputs, outputs, initializer=initializers)

        # Pin the opset, the default tracks the installed onnx package and can run ahead of onnxruntime
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
        model.ir_version = 9

        path = os.path.join(self.directory, name)
        onnx.save(model, path)

        return path

    def model(self, path, **config):
        """
        Creates an ONNX vectors instance.

        Args:
            path: model path
            config: additional configuration

        Returns:
            ONNX
        """

        return ONNX({"path": path, "tokenizer": self.tokenizer, "gpu": False, **config}, None, None)

    def testTokenOutput(self):
        """
        Test that token level output is pooled into a single vector per input
        """

        model = self.model(self.build("tokens.onnx"))
        self.assertEqual(model.encode(["dog puppy", "quantum"]).shape, (2, 4))

    def testPooledOutput(self):
        """
        Test that output which is already pooled passes through
        """

        model = self.model(self.build("pooled.onnx", pooled=True))
        self.assertEqual(model.encode(["dog puppy"]).shape, (1, 4))

    def testTokenTypeIds(self):
        """
        Test that token_type_ids is set only when the model declares it
        """

        self.assertEqual(self.model(self.build("types.onnx", types=True)).encode(["dog", "quantum physics"]).shape, (2, 4))
        self.assertEqual(self.model(self.build("notypes.onnx")).encode(["dog", "quantum physics"]).shape, (2, 4))

    def testPadding(self):
        """
        Test that padding is excluded when pooling
        """

        model = self.model(self.build("padding.onnx"))

        # An input padded to a longer sequence must produce the same vector as that input alone
        self.assertTrue(np.allclose(model.encode(["dog"])[0], model.encode(["dog", "quantum physics"])[0], atol=1e-6))

    def testMaxLength(self):
        """
        Test that maxlength truncates long inputs
        """

        model = self.model(self.build("maxlength.onnx"), maxlength=1)
        self.assertTrue(np.allclose(model.encode(["dog quantum physics"])[0], model.encode(["dog"])[0], atol=1e-6))

    def testTokenizerPath(self):
        """
        Test that a tokenizer stored alongside the model is found when one isn't configured
        """

        directory = os.path.join(self.directory, "sibling")
        os.makedirs(directory, exist_ok=True)
        shutil.copy(self.tokenizer, os.path.join(directory, "tokenizer.json"))

        path = self.build(os.path.join("sibling", "model.onnx"))
        model = ONNX({"path": path, "gpu": False}, None, None)

        self.assertEqual(model.encode(["dog"]).shape, (1, 4))

    def testProviders(self):
        """
        Test provider selection
        """

        model = self.model(self.build("providers.onnx"))

        with patch("txtai.vectors.dense.onnx.ort.get_available_providers") as providers:
            providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            model.config["gpu"] = True
            self.assertEqual(model.providers(), ["CUDAExecutionProvider", "CPUExecutionProvider"])

            model.config["gpu"] = False
            self.assertEqual(model.providers(), ["CPUExecutionProvider"])

            providers.return_value = ["CPUExecutionProvider"]
            model.config["gpu"] = True
            self.assertEqual(model.providers(), ["CPUExecutionProvider"])
