"""
LiteRT module tests
"""

import unittest

from txtai.pipeline import LLM


class TestLiteRT(unittest.TestCase):
    """
    LiteRT tests.
    """

    def testGeneration(self):
        """
        Test generation with LiteRT
        """

        # Test model generation with LiteRT
        model = LLM("neuml/gemma-4-tiny-random-litert-lm/gemma-4-tiny-random.litertlm", mtp=False, maxlength=25)

        # Test standard
        self.assertIsNotNone(model("Hello"))

        # Test streaming
        self.assertIsNotNone(list(model("Hello", stream=True)))

        # Test CPU fallback
        model = LLM("neuml/gemma-4-tiny-random-litert-lm/gemma-4-tiny-random.litertlm", mtp=True, maxlength=25)
        self.assertIsNotNone(model("Hello"))
