"""
Summary module tests
"""

import unittest

from txtai.pipeline import Textractor

# pylint: disable = C0411
from utils import Utils


class TestTextractor(unittest.TestCase):
    """
    Textractor tests.
    """

    def testBeautifulSoup(self):
        """
        Test text extraction using Beautiful Soup
        """

        textractor = Textractor(tika=False)
        text = textractor(Utils.PATH + "/tabular.csv")
        self.assertEqual(len(text), 125)

    def testCheckJava(self):
        """
        Test the checkjava method
        """

        textractor = Textractor()
        self.assertFalse(textractor.checkjava("1112444abc"))

    def testLines(self):
        """
        Test extraction to lines
        """

        textractor = Textractor(lines=True)

        # Extract text as lines
        lines = textractor(Utils.PATH + "/article.pdf")

        # Check number of lines is as expected
        self.assertEqual(len(lines), 35)

    def testParagraphs(self):
        """
        Test extraction to paragraphs
        """

        textractor = Textractor(paragraphs=True)

        # Extract text as sentences
        paragraphs = textractor(Utils.PATH + "/article.pdf")

        # Check number of paragraphs is as expected
        self.assertEqual(len(paragraphs), 13)

    def testSentences(self):
        """
        Test extraction to sentences
        """

        textractor = Textractor(sentences=True)

        # Extract text as sentences
        sentences = textractor(Utils.PATH + "/article.pdf")

        # Check number of sentences is as expected
        self.assertEqual(len(sentences), 17)

    def testSingle(self):
        """
        Test a single extraction with no tokenization of the results
        """

        textractor = Textractor()

        # Extract text as a single block
        text = textractor(Utils.PATH + "/article.pdf")

        # Check length of text is as expected
        self.assertEqual(len(text), 2301)
