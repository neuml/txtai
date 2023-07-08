"""
Generator module tests
"""

import unittest

from txtai.pipeline import Generator


class TestGenerator(unittest.TestCase):
    """
    Sequences tests.
    """

    def testGeneration(self):
        """
        Test text pipeline generation
        """

        model = Generator("hf-internal-testing/tiny-random-gpt2")
        start = "Hello, how are"

        # Test that text is generated
        self.assertIsNotNone(model(start))
