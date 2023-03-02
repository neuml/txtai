"""
Embeddings+database module tests
"""

import contextlib
import io
import os
import tempfile
import unittest

from txtai.embeddings import Embeddings
from txtai.database import Database, SQLError


# pylint: disable=R0904
class TestEmbeddings(unittest.TestCase):
    """
    Embeddings with a database tests.
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

    @classmethod
    def tearDownClass(cls):
        """
        Cleanup data.
        """

        if cls.embeddings:
            cls.embeddings.close()

    def testArchive(self):
        """
        Test embeddings index archiving
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
        Test embeddings close
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

    def testEmpty(self):
        """
        Test empty index
        """

        # Test search against empty index
        embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": True})
        self.assertEqual(embeddings.search("test"), [])

        # Test index with no data
        embeddings.index([])
        self.assertIsNone(embeddings.ann)

        # Test upsert with no data
        embeddings.index([(0, "this is a test", None)])
        embeddings.upsert([])
        self.assertIsNotNone(embeddings.ann)

    def testExplain(self):
        """
        Test query explain
        """

        # Test explain with similarity
        result = self.embeddings.explain("feel good story", self.data)[0]
        self.assertEqual(result["text"], self.data[4])
        self.assertEqual(len(result.get("tokens")), 8)

    def testExplainBatch(self):
        """
        Test query explain batch
        """

        # Test explain with query
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        result = self.embeddings.batchexplain(["feel good story"], limit=1)[0][0]
        self.assertEqual(result["text"], self.data[4])
        self.assertEqual(len(result.get("tokens")), 8)

    def testExplainEmpty(self):
        """
        Test query explain with no filtering criteria
        """

        self.assertEqual(self.embeddings.explain("select * from txtai limit 1")[0]["id"], "0")

    def testFunction(self):
        """
        Test custom functions
        """

        embeddings = Embeddings(
            {
                "path": "sentence-transformers/nli-mpnet-base-v2",
                "content": True,
                "functions": [{"name": "length", "function": "testdatabase.testembeddings.length"}],
            }
        )

        # Create an index for the list of text
        embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Search for best match
        result = embeddings.search("select length(text) length from txtai where id = 0", 1)[0]

        self.assertEqual(result["length"], 39)

    def testGenerator(self):
        """
        Test index with a generator
        """

        def documents():
            for uid, text in enumerate(self.data):
                yield (uid, text, None)

        # Create an index for the list of text
        self.embeddings.index(documents())

        # Search for best match
        result = self.embeddings.search("feel good story", 1)[0]

        self.assertEqual(result["text"], self.data[4])

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

    def testInfo(self):
        """
        Test info
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.embeddings.info()

        self.assertIn("txtai", output.getvalue())

    def testInstructions(self):
        """
        Test indexing with instruction prefixes.
        """

        embeddings = Embeddings(
            {"path": "sentence-transformers/nli-mpnet-base-v2", "content": True, "instructions": {"query": "query: ", "data": "passage: "}}
        )

        embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Search for best match
        result = embeddings.search("feel good story", 1)[0]

        self.assertEqual(result["text"], self.data[4])

    def testInvalidData(self):
        """
        Test invalid JSON data
        """

        # Test invalid JSON value
        with self.assertRaises(ValueError):
            self.embeddings.index([(0, {"text": "This is a test", "flag": float("NaN")}, None)])

    def testJSON(self):
        """
        Test JSON configuration
        """

        embeddings = Embeddings(
            {
                "format": "json",
                "path": "sentence-transformers/nli-mpnet-base-v2",
                "content": True,
            }
        )

        embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "embeddings.json")

        embeddings.save(index)

        # Check that config.json exists
        self.assertTrue(os.path.exists(os.path.join(index, "config.json")))

        # Check that index can be reloaded
        embeddings.load(index)
        self.assertEqual(embeddings.count(), 6)

    def testMultiData(self):
        """
        Test indexing with multiple data types (text, documents)
        """

        embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": True, "batch": len(self.data)})

        # Create an index using mixed data (text and documents)
        data = []
        for uid, text in enumerate(self.data):
            data.append((uid, text, None))
            data.append((uid, {"content": text}, None))

        embeddings.index(data)

        # Search for best match
        result = embeddings.search("feel good story", 1)[0]

        self.assertEqual(result["text"], self.data[4])

    def testMultiSave(self):
        """
        Test multiple successive saves
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
        Test exceptions for non-implemented methods
        """

        database = Database({})

        self.assertRaises(NotImplementedError, database.load, None)
        self.assertRaises(NotImplementedError, database.insert, None)
        self.assertRaises(NotImplementedError, database.delete, None)
        self.assertRaises(NotImplementedError, database.reindex, None)
        self.assertRaises(NotImplementedError, database.save, None)
        self.assertRaises(NotImplementedError, database.close)
        self.assertRaises(NotImplementedError, database.ids, None)
        self.assertRaises(NotImplementedError, database.resolve, None, None)
        self.assertRaises(NotImplementedError, database.embed, None, None)
        self.assertRaises(NotImplementedError, database.query, None, None)

    def testQueryModel(self):
        """
        Test index
        """

        embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": True, "query": {"path": "neuml/t5-small-txtsql"}})

        # Create an index for the list of text
        embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Search for best match
        result = embeddings.search("feel good story with maine in text", 1)[0]

        self.assertEqual(result["text"], self.data[4])

    def testReindex(self):
        """
        Test reindex
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Delete records to test indexids still match
        self.embeddings.delete(([0, 1]))

        # Reindex
        self.embeddings.reindex({"path": "sentence-transformers/nli-mpnet-base-v2"}, ["text"])

        # Search for best match
        result = self.embeddings.search("feel good story", 1)[0]

        self.assertEqual(result["text"], self.data[4])

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
        self.embeddings.index([(uid, {"text": text, "length": len(text)}, None) for uid, text in enumerate(self.data)])

        # Test similar
        result = self.embeddings.search(
            "select * from txtai where similar('feel good story') group by text having count(*) > 0 order by score desc", 1
        )[0]
        self.assertEqual(result["text"], self.data[4])

        # Test similar with limits
        result = self.embeddings.search("select * from txtai where similar('feel good story', 1) limit 1")[0]
        self.assertEqual(result["text"], self.data[4])

        # Test similar with offset
        result = self.embeddings.search("select * from txtai where similar('feel good story') offset 1")[0]
        self.assertEqual(result["text"], self.data[5])

        # Test where
        result = self.embeddings.search("select * from txtai where text like '%iceberg%'", 1)[0]
        self.assertEqual(result["text"], self.data[1])

        # Test count
        result = self.embeddings.search("select count(*) from txtai")[0]
        self.assertEqual(list(result.values())[0], len(self.data))

        # Test columns
        result = self.embeddings.search("select id, text, length, data, entry from txtai")[0]
        self.assertEqual(sorted(result.keys()), ["data", "entry", "id", "length", "text"])

        # Test SQL parse error
        with self.assertRaises(SQLError):
            self.embeddings.search("select * from txtai where bad,query")

    def testTerms(self):
        """
        Test extracting keyword terms from query
        """

        result = self.embeddings.terms("select * from txtai where similar('keyword terms')")
        self.assertEqual(result, "keyword terms")

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

    def testUpsertBatch(self):
        """
        Test upsert batch
        """

        try:
            # Build data array
            data = [(uid, text, None) for uid, text in enumerate(self.data)]

            # Reset embeddings for test
            self.embeddings.ann = None

            # Create an index for the list of text
            self.embeddings.upsert(data)

            # Set batch size to 1
            self.embeddings.config["batch"] = 1

            # Update data
            data[0] = (0, "Feel good story: baby panda born", None)
            data[1] = (0, "Not good news", None)
            self.embeddings.upsert([data[0], data[1]])

            # Search for best match
            result = self.embeddings.search("feel good story", 1)[0]

            self.assertEqual(result["text"], data[0][1])
        finally:
            del self.embeddings.config["batch"]


def length(text):
    """
    Custom SQL function.
    """

    return len(text)
