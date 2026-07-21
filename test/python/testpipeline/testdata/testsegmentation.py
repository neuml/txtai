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
        Test paragraph segmentation with both LF and CRLF blank lines
        """

        segment = Segmentation(paragraphs=True)
        self.assertEqual(segment("Para A.\n\nPara B."), ["Para A.", "Para B."])

        # Windows blank lines (\r\n\r\n) must split the same way as LF.
        self.assertEqual(segment("Para A.\r\n\r\nPara B."), ["Para A.", "Para B."])

    def testSections(self):
        """
        Test section segmentation with both LF and CRLF blank lines
        """

        segment = Segmentation(sections=True)

        self.assertEqual(segment("Sec A.\n\n\nSec B."), ["Sec A.", "Sec B."])

        self.assertEqual(
            segment("Sec A.\r\n\r\n\r\nSec B.\r\n\r\n\r\nSec C."),
            ["Sec A.", "Sec B.", "Sec C."],
        )
