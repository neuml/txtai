"""
Tabular module tests
"""

import unittest

from txtai.pipeline import Tabular

# pylint: disable=C0411
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

    def testFalsyIDColumn(self):
        """
        Explicit column labels are used for IDs and excluded from default text.
        """

        for column in ("", 0, "id"):
            with self.subTest(column=column):
                rows = Tabular(idcolumn=column)([{column: "doc-7", "text": "hello"}])
                self.assertEqual(rows, [("doc-7", "hello", None)])

    def testFalsyValues(self):
        """
        Test that zero and False are indexed as content
        """

        tabular = Tabular("id", ["quantity", "text"])

        # A quantity of 0 is real data. Before this was fixed, the truthy check in concat
        # dropped it and the row indexed as "Widget", losing the quantity.
        rows = tabular([{"id": 1, "quantity": 0, "text": "Widget"}])
        self.assertEqual(rows[0][1], "0. Widget")

        rows = tabular([{"id": 2, "quantity": False, "text": "Widget"}])
        self.assertEqual(rows[0][1], "False. Widget")

    def testInvalid(self):
        """
        Test invalid file paths
        """

        with self.assertRaises(ValueError):
            self.tabular([Utils.PATH + "/article.pdf"])

        with self.assertRaises(ValueError):
            self.tabular(["https://invalid.path"])

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

    def testMissingIDFallsBack(self):
        """
        Rows without an ID (missing key, None or NaN) fall back to the row index.
        pandas coerces the missing value to NaN, which previously bypassed the
        is-not-None fallback and indexed rows with a NaN uid.
        """

        tabular = Tabular("id", ["text"])

        rows = tabular(
            [
                {"id": "a", "text": "first"},
                {"text": "second"},
                {"id": None, "text": "third"},
                {"id": float("nan"), "text": "fourth"},
                {"id": "c", "text": "fifth"},
            ]
        )
        self.assertEqual([row[0] for row in rows], ["a", 1, 2, 3, "c"])

    def testNoColumns(self):
        """
        Test creating text without specifying columns
        """

        tabular = Tabular("id")
        rows = tabular([{"id": 0, "text": "This is a test", "summary": "Describes text in more detail"}])
        uid, text, _ = rows[0]

        self.assertEqual(uid, 0)
        self.assertEqual(text, "This is a test. Describes text in more detail")

    def testNullValues(self):
        """
        Test that null and empty values are still skipped
        """

        tabular = Tabular("id", ["quantity", "text"])

        # NaN is the null marker column() normalizes to None, and an empty string adds
        # nothing but a separator. Both stay excluded.
        rows = tabular([{"id": 1, "quantity": float("nan"), "text": "Widget"}])
        self.assertEqual(rows[0][1], "Widget")

        rows = tabular([{"id": 2, "quantity": "", "text": "Widget"}])
        self.assertEqual(rows[0][1], "Widget")

    def testNumericTypes(self):
        """
        Mixed numeric rows preserve integer IDs and content without float rounding.
        """

        data = [{"id": 2**53 + 1, "count": 2, "value": 1.5}, {"id": 2**53, "count": 3, "value": 2.5}]
        for content in (False, True, ["id"]):
            with self.subTest(content=content):
                rows = Tabular("id", ["count", "value"], content)(data)
                step = 2 if content else 1
                self.assertTrue(all(isinstance(row[0], int) for row in rows))
                self.assertEqual([row[0] for row in rows[::step]], [row["id"] for row in data])
                self.assertEqual([row[1] for row in rows[::step]], ["2. 1.5", "3. 2.5"])
                if content:
                    expected = data if content is True else [{"id": row["id"]} for row in data]
                    self.assertEqual([row[1] for row in rows[1::2]], expected)
