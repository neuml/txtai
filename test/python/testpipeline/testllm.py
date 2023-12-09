"""
LLM module tests
"""

import unittest

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from txtai.pipeline import LLM, Generation


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

    def testCustom(self):
        """
        Test custom LLM framework
        """

        model = LLM("hf-internal-testing/tiny-random-gpt2", task="language-generation", method="txtai.pipeline.HFGeneration")
        self.assertIsNotNone(model("Hello, how are"))

    def testCustomNotFound(self):
        """
        Test resolving an unresolvable LLM framework
        """

        with self.assertRaises(ImportError):
            LLM("hf-internal-testing/tiny-random-gpt2", method="notfound.generation")

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

    def testNotImplemented(self):
        """
        Test exceptions for non-implemented methods
        """

        generation = Generation()
        self.assertRaises(NotImplementedError, generation.execute, None, None)
