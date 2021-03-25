"""
Summary module tests
"""

import unittest

from txtai.pipeline import Textractor

# pylint: disable = C0411
from utils import Utils


class TestSummary(unittest.TestCase):
    """
    Labels tests
    """

    def testSingle(self):
        """
        Tests a single extraction with no tokenization of the results
        """

        textractor = Textractor()

        # Extract text as a single block
        text = textractor(Utils.PATH + "/article.pdf")

        # Check length of text is as expected
        self.assertEqual(len(text), 2301)

    def testSentences(self):
        """
        Tests extraction to sentences
        """

        textractor = Textractor(sentences=True)

        # Extract text as sentences
        sentences = textractor(Utils.PATH + "/article.pdf")

        # Check number of sentences is as expected
        self.assertEqual(len(sentences), 17)

    def testParagraphs(self):
        """
        Tests extraction to paragraphs
        """

        textractor = Textractor(paragraphs=True)

        # Extract text as sentences
        paragraphs = textractor(Utils.PATH + "/article.pdf")

        # Check number of paragraphs is as expected
        self.assertEqual(len(paragraphs), 13)
