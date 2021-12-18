"""
Embeddings+database module tests
"""

import os
import tempfile
import unittest

from txtai.embeddings import Embeddings
from txtai.database import Database, SQLException


class TestEmbeddings(unittest.TestCase):
    """
    Embeddings with a database tests
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

        # Create embeddings model, backed by sentence-transformers & transformers
        cls.embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": True})

    def testArchive(self):
        """
        Tests embeddings index archiving
        """

        for extension in ["tar.bz2", "tar.gz", "tar.xz", "zip"]:
            # Create an index for the list of text
            self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

            # Generate temp file path
            index = os.path.join(tempfile.gettempdir(), f"embeddings.{extension}")

            self.embeddings.save(index)
            self.embeddings.load(index)

            # Search for best match
            result = self.embeddings.search("feel good story", 1)[0]

            self.assertEqual(result["text"], self.data[4])

            # Test offsets still work after save/load
            self.embeddings.upsert([(0, "Looking out into the dreadful abyss", None)])
            self.assertEqual(self.embeddings.count(), len(self.data))

    def testClose(self):
        """
        Tests embeddings close
        """

        embeddings = None

        # Create index twice to test open/close and ensure resources are freed
        for _ in range(2):
            embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": True})

            # Add record to index
            embeddings.index([(0, "Close test", None)])

            # Save index
            index = os.path.join(tempfile.gettempdir(), "embeddings.close")
            embeddings.save(index)

            # Close index
            embeddings.close()

        # Test embeddings is empty
        self.assertIsNone(embeddings.ann)
        self.assertIsNone(embeddings.database)

    def testData(self):
        """
        Test content storage and retrieval
        """

        data = self.data + [{"date": "2021-01-01", "text": "Baby panda", "flag": 1}]

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(data)])

        # Search for best match
        result = self.embeddings.search("feel good story", 1)[0]
        self.assertEqual(result["text"], data[-1]["text"])

    def testDelete(self):
        """
        Test delete
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Delete best match
        self.embeddings.delete([4])

        # Search for best match
        result = self.embeddings.search("feel good story", 1)[0]

        self.assertEqual(self.embeddings.count(), 5)
        self.assertEqual(result["text"], self.data[5])

    def testIndex(self):
        """
        Test index
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Search for best match
        result = self.embeddings.search("feel good story", 1)[0]

        self.assertEqual(result["text"], self.data[4])

    def testIndexTokens(self):
        """
        Test index with tokens
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text.split(), None) for uid, text in enumerate(self.data)])

        # Search for best match
        result = self.embeddings.search("feel good story", 1)[0]

        self.assertEqual(result["text"], self.data[4])

    def testMultiSave(self):
        """
        Tests multiple successive saves
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Save original index
        index = os.path.join(tempfile.gettempdir(), "embeddings.insert")
        self.embeddings.save(index)

        # Modify index
        self.embeddings.upsert([(0, "Looking out into the dreadful abyss", None)])

        # Save to a different location
        indexupdate = os.path.join(tempfile.gettempdir(), "embeddings.update")
        self.embeddings.save(indexupdate)

        # Save to same location
        self.embeddings.save(index)

        # Test all indexes match
        result = self.embeddings.search("feel good story", 1)[0]
        self.assertEqual(result["text"], self.data[4])

        self.embeddings.load(index)
        result = self.embeddings.search("feel good story", 1)[0]
        self.assertEqual(result["text"], self.data[4])

        self.embeddings.load(indexupdate)
        result = self.embeddings.search("feel good story", 1)[0]
        self.assertEqual(result["text"], self.data[4])

    def testNotImplemented(self):
        """
        Tests exceptions for non-implemented methods
        """

        database = Database({})

        self.assertRaises(NotImplementedError, database.load, None)
        self.assertRaises(NotImplementedError, database.insert, None)
        self.assertRaises(NotImplementedError, database.delete, None)
        self.assertRaises(NotImplementedError, database.save, None)
        self.assertRaises(NotImplementedError, database.close)
        self.assertRaises(NotImplementedError, database.ids, None)
        self.assertRaises(NotImplementedError, database.resolve, None, None)
        self.assertRaises(NotImplementedError, database.embed, None, None)
        self.assertRaises(NotImplementedError, database.query, None, None)

    def testSave(self):
        """
        Test save
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "embeddings")

        self.embeddings.save(index)
        self.embeddings.load(index)

        # Search for best match
        result = self.embeddings.search("feel good story", 1)[0]

        self.assertEqual(result["text"], self.data[4])

        # Test offsets still work after save/load
        self.embeddings.upsert([(0, "Looking out into the dreadful abyss", None)])
        self.assertEqual(self.embeddings.count(), len(self.data))

    def testSQL(self):
        """
        Test running a SQL query
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Test similar
        result = self.embeddings.search(
            "select * from txtai where similar('feel good story') group by text having count(*) > 0 order by score desc", 1
        )[0]
        self.assertEqual(result["text"], self.data[4])

        # Test similar with limits
        result = self.embeddings.search("select * from txtai where similar('feel good story', 1) limit 1")[0]
        self.assertEqual(result["text"], self.data[4])

        # Test where
        result = self.embeddings.search("select * from txtai where text like '%iceberg%'", 1)[0]
        self.assertEqual(result["text"], self.data[1])

        # Test count
        result = self.embeddings.search("select count(*) from txtai")[0]
        self.assertEqual(list(result.values())[0], len(self.data))

        # Test columns
        result = self.embeddings.search("select id, text, data, entry from txtai")[0]
        self.assertEqual(sorted(result.keys()), ["data", "entry", "id", "text"])

        # Test SQL parse error
        with self.assertRaises(SQLException):
            self.embeddings.search("select * from txtai where bad,query")

    def testUpsert(self):
        """
        Test upsert
        """

        # Build data array
        data = [(uid, text, None) for uid, text in enumerate(self.data)]

        # Reset embeddings for test
        self.embeddings.ann = None

        # Create an index for the list of text
        self.embeddings.upsert(data)

        # Update data
        data[0] = (0, "Feel good story: baby panda born", None)
        self.embeddings.upsert([data[0]])

        # Search for best match
        result = self.embeddings.search("feel good story", 1)[0]

        self.assertEqual(result["text"], data[0][1])
