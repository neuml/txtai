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
        Test Tokenizer.tokenize
        """

        self.assertEqual(Tokenizer.tokenize("this is a test"), ["test"])
        self.assertEqual(Tokenizer.tokenize("abc123 abc 123"), ["abc123", "abc"])
