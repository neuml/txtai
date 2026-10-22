"""
Segmentation module tests
"""

import unittest

from txtai.pipeline import Segmentation


class TestSegmentation(unittest.TestCase):
    """
    Segmentation tests.
    """

    def testLines(self):
        """
        Test line segmentation tolerates CRLF
        """

        segment = Segmentation(lines=True)
        self.assertEqual(segment("Line A.\r\nLine B."), ["Line A.", "Line B."])

    def testParagraphs(self):
        """
        Test paragraph segmentation with LF, CRLF and whitespace-only blank lines
        """

        segment = Segmentation(paragraphs=True)
        self.assertEqual(segment("Para A.\n\nPara B."), ["Para A.", "Para B."])

        # Windows blank lines (\r\n\r\n) must split the same way as LF.
        self.assertEqual(segment("Para A.\r\n\r\nPara B."), ["Para A.", "Para B."])

        # Blank lines that only contain spaces or tabs must split the same way.
        self.assertEqual(segment("Para A.\n  \nPara B."), ["Para A.", "Para B."])
        self.assertEqual(segment("Para A.\r\n\t\r\nPara B."), ["Para A.", "Para B."])

        # Whitespace inside a paragraph is kept.
        self.assertEqual(segment("Line A.\n  Line B.\n\nPara B."), ["Line A.\n Line B.", "Para B."])

    def testSections(self):
        """
        Test section segmentation with LF, CRLF and whitespace-only blank lines
        """

        segment = Segmentation(sections=True)

        self.assertEqual(segment("Sec A.\n\n\nSec B."), ["Sec A.", "Sec B."])

        self.assertEqual(
            segment("Sec A.\r\n\r\n\r\nSec B.\r\n\r\n\r\nSec C."),
            ["Sec A.", "Sec B.", "Sec C."],
        )

        # Blank lines that only contain spaces or tabs must split the same way.
        self.assertEqual(segment("Sec A.\n \n\t\nSec B."), ["Sec A.", "Sec B."])
