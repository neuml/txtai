"""
LLM module tests
"""

import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer

from txtai.pipeline import LLM


class TestLLM(unittest.TestCase):
    """
    LLM tests.
    """

    def testExternal(self):
        """
        Test externally loaded model
        """

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        model = LLM((model, tokenizer))
        start = "Hello, how are"

        # Test that text is generator
        self.assertGreater(len(model(start)), len(start))
