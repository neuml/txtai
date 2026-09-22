"""
DuckDB module tests
"""

import os
import unittest

from txtai.database.factory import DatabaseFactory
from txtai.embeddings import Embeddings

from .testrdbms import Common


# pylint: disable=R0904
class TestDuckDB(Common.TestRDBMS):
    """
    Embeddings with content stored in DuckDB.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize test data.
        """

        cls.data = [
            "US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day",
        ]

        # Content backend
        cls.backend = "duckdb"

        # Create embeddings model, backed by sentence-transformers & transformers
        cls.embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": cls.backend})

    @classmethod
    def tearDownClass(cls):
        """
        Cleanup data.
        """

        if cls.embeddings:
            cls.embeddings.close()

    @unittest.skipIf(os.name == "nt", "testArchive skipped on Windows")
    def testArchive(self):
        """
        Test embeddings index archiving
        """

        super().testArchive()

    def testFunction(self):
        """
        Test custom functions
        """

        embeddings = Embeddings(
            {
                "path": "sentence-transformers/nli-mpnet-base-v2",
                "content": self.backend,
                "functions": [{"name": "textlength", "function": "testdatabase.testduckdb.length"}],
            }
        )

        # Create an index for the list of text
        embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Search for best match
        result = embeddings.search("select textlength(text) length from txtai where id = 0", 1)[0]

        self.assertEqual(int(result["length"]), 39)

    def testRepeatedBindParameter(self):
        """
        Test a query that references the same named bind parameter more than once
        """

        database = DatabaseFactory.create({"content": self.backend})
        database.insert([(0, "a fully intact ice shelf collapsed, forming an iceberg", None)])

        result = database.search("select * from txtai where text like :x or text like :x", parameters={"x": "%iceberg%"})
        self.assertEqual(len(result), 1)

        database.close()


def length(text):
    """
    Custom SQL function.
    """

    return len(text)
