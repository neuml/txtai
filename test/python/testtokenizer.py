"""
Tokenizer module tests
"""

import unittest

from txtai.tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):
    """
    Tokenizer tests
    """

    def testTokenize(self):
        """
        Test tokenize
        """

        self.assertEqual(Tokenizer.tokenize("Y this is a test!"), ["test"])
        self.assertEqual(Tokenizer.tokenize("abc123 ABC 123"), ["abc123", "abc"])
