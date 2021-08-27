"""
ONNX module tests
"""

import os
import tempfile
import unittest

import numpy as np

from onnxruntime import InferenceSession, SessionOptions

from transformers import AutoTokenizer

from txtai.pipeline import HFOnnx, HFTrainer


class TestOnnx(unittest.TestCase):
    """
    ONNX tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Create default datasets
        """

        cls.data = [{"text": "Dogs", "label": 0}, {"text": "dog", "label": 0}, {"text": "Cats", "label": 1}, {"text": "cat", "label": 1}] * 100

    def testDefault(self):
        """
        Test exporting an ONNX model with default parameters
        """

        # Export model to ONNX, use default parameters
        onnx = HFOnnx()
        model = onnx("google/bert_uncased_L-2_H-128_A-2")

        # Validate model has data
        self.assertGreater(len(model), 0)

    def testClassification(self):
        """
        Test exporting a classification model to ONNX and running inference
        """

        # Normalize logits using sigmoid function
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

        trainer = HFTrainer()
        model, tokenizer = trainer("google/bert_uncased_L-2_H-128_A-2", self.data)

        # Output file path
        output = os.path.join(tempfile.gettempdir(), "onnx")

        # Export model to ONNX
        onnx = HFOnnx()
        model = onnx((model, tokenizer), "sequence-classification", output, True)

        # Build ONNX session
        options = SessionOptions()
        session = InferenceSession(model, options)

        # Tokenize and cast to int64 to support all platforms
        tokens = tokenizer(["cat"], return_tensors="np")
        tokens = {x: tokens[x].astype(np.int64) for x in tokens}

        # Run inference and validate
        outputs = session.run(None, dict(tokens))
        outputs = sigmoid(outputs[0])
        self.assertEqual(np.argmax(outputs[0]), 1)

    def testPooling(self):
        """
        Test exporting a pooling model to ONNX and running inference
        """

        # Export model to ONNX
        onnx = HFOnnx()
        model = onnx("sentence-transformers/paraphrase-MiniLM-L3-v2", "pooling", quantize=True)
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")

        # Build ONNX session
        options = SessionOptions()
        session = InferenceSession(model, options)

        # Tokenize and cast to int64 to support all platforms
        tokens = tokenizer(["cat"], return_tensors="np")
        tokens = {x: tokens[x].astype(np.int64) for x in tokens}

        # Run inference and validate
        outputs = session.run(None, dict(tokens))
        self.assertEqual(outputs[0].shape, (1, 384))
