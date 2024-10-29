"""
Tabular module tests
"""

import unittest

from txtai.pipeline import Tabular

# pylint: disable = C0411
from utils import Utils


class TestTabular(unittest.TestCase):
    """
    Tabular tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create single tabular instance
        """

        cls.tabular = Tabular("id", ["text"])

    def testContent(self):
        """
        Test parsing additional content
        """

        tabular = Tabular("id", ["text"], True)

        row = {"id": 0, "text": "This is a test", "flag": 1}

        # When content is enabled, both (uid, text, tags) and (uid, data, tags) rows are generated
        # given that data doesn't necessarily include the text to index
        rows = tabular([row])
        uid, data, _ = rows[1]

        # Data should contain the entire input row
        self.assertEqual(uid, 0)
        self.assertEqual(data, row)

        # Only select flag field
        tabular.content = ["flag"]
        rows = tabular([row])
        uid, data, _ = rows[1]

        # Data should only contain a single field, flag
        self.assertEqual(uid, 0)
        self.assertTrue(list(data.keys()) == ["flag"])
        self.assertEqual(data["flag"], 1)

    def testCSV(self):
        """
        Test parsing a CSV file
        """

        rows = self.tabular([Utils.PATH + "/tabular.csv"])
        uid, text, _ = rows[0][0]

        self.assertEqual(uid, 0)
        self.assertEqual(text, "The first sentence")

    def testDict(self):
        """
        Test parsing a dict
        """

        rows = self.tabular([{"id": 0, "text": "This is a test"}])
        uid, text, _ = rows[0]

        self.assertEqual(uid, 0)
        self.assertEqual(text, "This is a test")

    def testList(self):
        """
        Test parsing a list
        """

        rows = self.tabular([[{"id": 0, "text": "This is a test"}]])
        uid, text, _ = rows[0][0]

        self.assertEqual(uid, 0)
        self.assertEqual(text, "This is a test")

    def testMissingColumns(self):
        """
        Test rows with uneven or missing columns
        """

        tabular = Tabular("id", ["text"], True)

        rows = tabular([{"id": 0, "text": "This is a test", "metadata": "meta"}, {"id": 1, "text": "This is a test"}])

        # When content is enabled both (id, text, tag) and (id, data, tag) tuples are generated given that
        # data doesn't necessarily include the text to index
        _, data, _ = rows[3]

        self.assertIsNone(data["metadata"])

    def testNoColumns(self):
        """
        Test creating text without specifying columns
        """

        tabular = Tabular("id")
        rows = tabular([{"id": 0, "text": "This is a test", "summary": "Describes text in more detail"}])
        uid, text, _ = rows[0]

        self.assertEqual(uid, 0)
        self.assertEqual(text, "This is a test. Describes text in more detail")
