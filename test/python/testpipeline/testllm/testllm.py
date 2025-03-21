"""
LLM module tests
"""

import unittest

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from txtai.pipeline import LLM, Generation

# pylint: disable = C0411
from utils import Utils


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

    def testBatchSize(self):
        """
        Test batch size
        """

        model = LLM("sshleifer/tiny-gpt2")
        self.assertIsNotNone(model(["Hello, how are"] * 2, batch_size=2))

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

    def testDefaultRole(self):
        """
        Test default role
        """

        model = LLM("hf-internal-testing/tiny-random-LlamaForCausalLM")
        self.assertIsNotNone(model("Hello, how are", defaultrole="user"))

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

    def testMaxLength(self):
        """
        Test max length
        """

        model = LLM("sshleifer/tiny-gpt2")
        self.assertIsInstance(model("Hello, how are", maxlength=10), str)

    def testNotImplemented(self):
        """
        Test exceptions for non-implemented methods
        """

        generation = Generation()
        self.assertRaises(NotImplementedError, generation.stream, None, None, None, None)

    def testStop(self):
        """
        Test stop strings
        """

        model = LLM("sshleifer/tiny-gpt2")
        self.assertIsNotNone(model("Hello, how are", stop=["you"]))

    def testStream(self):
        """
        Test streaming generation
        """

        model = LLM("sshleifer/tiny-gpt2")
        self.assertIsInstance(" ".join(x for x in model("Hello, how are", stream=True)), str)

    def testVision(self):
        """
        Test vision LLM
        """

        model = LLM("neuml/tiny-random-qwen2vl")
        result = model(
            [{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image", "image": Utils.PATH + "/books.jpg"}]}]
        )

        self.assertIsNotNone(result)
