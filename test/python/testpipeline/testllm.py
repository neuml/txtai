"""
LLM module tests
"""

import unittest

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from txtai.pipeline import LLM


class TestLLM(unittest.TestCase):
    """
    LLM tests.
    """

    def testArguments(self):
        """
        Test pipeline keyword arguments
        """

        start = "Hello, how are"

        # Test that text is generated with custom parameters
        model = LLM("hf-internal-testing/tiny-random-gpt2", task="language-generation", torch_dtype="torch.float32")
        self.assertIsNotNone(model(start))

        model = LLM("hf-internal-testing/tiny-random-gpt2", task="language-generation", torch_dtype=torch.float32)
        self.assertIsNotNone(model(start))

    def testExternal(self):
        """
        Test externally loaded model
        """

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        model = LLM((model, tokenizer), template="{text}")
        start = "Hello, how are"

        # Test that text is generated
        self.assertIsNotNone(model(start))
