"""
Tabular module tests
"""

import unittest

from txtai.pipeline import Tabular

# pylint: disable = C0411
from utils import Utils


class TestTabular(unittest.TestCase):
    """
    Tabular tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Create single tabular instance
        """

        cls.tabular = Tabular("id", ["text"])

    def testCSV(self):
        """
        Tests parsing a CSV file
        """

        rows = self.tabular([Utils.PATH + "/tabular.csv"])
        uid, text, _ = rows[0][0]

        self.assertEqual(uid, 0)
        self.assertEqual(text, "The first sentence")

    def testDict(self):
        """
        Tests parsing a dict
        """

        rows = self.tabular([{"id": 0, "text": "This is a test"}])
        uid, text, _ = rows[0]

        self.assertEqual(uid, 0)
        self.assertEqual(text, "This is a test")

    def testList(self):
        """
        Tests parsing a list
        """

        rows = self.tabular([[{"id": 0, "text": "This is a test"}]])
        uid, text, _ = rows[0][0]

        self.assertEqual(uid, 0)
        self.assertEqual(text, "This is a test")

    def testNoColumns(self):
        """
        Tests creating text without specifying columns
        """

        tabular = Tabular("id")
        rows = tabular([{"id": 0, "text": "This is a test", "summary": "Describes text in more detail"}])
        uid, text, _ = rows[0]

        self.assertEqual(uid, 0)
        self.assertEqual(text, "This is a test. Describes text in more detail")

    def testXLSX(self):
        """
        Tests parsing a XLSX file
        """

        rows = self.tabular([Utils.PATH + "/tabular.xlsx"])
        uid, text, _ = rows[0][0]

        self.assertEqual(uid, 0)
        self.assertEqual(text, "The first sentence")
