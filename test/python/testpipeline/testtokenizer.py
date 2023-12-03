"""
Tokenizer module tests
"""

import unittest

from txtai.pipeline import Tokenizer


class TestTokenizer(unittest.TestCase):
    """
    Tokenizer tests.
    """

    def testAlphanumTokenize(self):
        """
        Test alphanumeric tokenization
        """

        # Alphanumeric tokenization through backwards compatible static method
        self.assertEqual(Tokenizer.tokenize("Y this is a test!"), ["test"])
        self.assertEqual(Tokenizer.tokenize("abc123 ABC 123"), ["abc123", "abc"])

    def testEmptyTokenize(self):
        """
        Test handling empty and None inputs
        """

        # Test that parser can handle empty or None strings
        self.assertEqual(Tokenizer.tokenize(""), [])
        self.assertEqual(Tokenizer.tokenize(None), None)

    def testStandardTokenize(self):
        """
        Test standard tokenization
        """

        # Default standard tokenizer parameters
        tokenizer = Tokenizer()

        # Define token tests
        tests = [
            ("Y this is a test!", ["y", "this", "is", "a", "test"]),
            ("abc123 ABC 123", ["abc123", "abc", "123"]),
            ("Testing hy-phenated words", ["testing", "hy", "phenated", "words"]),
            ("111-111-1111", ["111", "111", "1111"]),
            ("Test.1234", ["test", "1234"]),
        ]

        # Run through tests
        for test, result in tests:
            # Unicode Text Segmentation per Unicode Annex #29
            self.assertEqual(tokenizer(test), result)
