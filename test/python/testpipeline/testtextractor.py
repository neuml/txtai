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

        # Extract text as paragraphs
        paragraphs = textractor(Utils.PATH + "/article.pdf")

        # Check number of paragraphs is as expected
        self.assertEqual(len(paragraphs), 11)

    def testSections(self):
        """
        Test extraction to sections
        """

        textractor = Textractor(sections=True)

        # Extract as sections
        paragraphs = textractor(Utils.PATH + "/document.pdf")

        # Check number of sections is as expected
        self.assertEqual(len(paragraphs), 3)

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
        self.assertEqual(len(text), 2333)

    def testTable(self):
        """
        Test table extraction
        """

        textractor = Textractor()

        # Extract text as a single block
        for name in ["document.docx", "spreadsheet.xlsx"]:
            text = textractor(f"{Utils.PATH}/{name}")

            # Check for table header
            self.assertTrue("|---|" in text)
