"""
Common file database module tests
"""

import contextlib
import io
import os
import tempfile
import unittest

from txtai.embeddings import Embeddings, IndexNotFoundError
from txtai.database import Embedded, RDBMS, SQLError


class Common:
    """
    Wraps common file database tests to prevent unit test discovery for this class.
    """

    # pylint: disable=R0904
    class TestRDBMS(unittest.TestCase):
        """
        Embeddings with content stored in a file database tests.
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
            cls.backend = None

            # Create embeddings model, backed by sentence-transformers & transformers
            cls.embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": cls.backend})

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
                index = os.path.join(tempfile.gettempdir(), f"embeddings.{self.category()}.{extension}")

                self.embeddings.save(index)
                self.embeddings.load(index)

                # Search for best match
                result = self.embeddings.search("feel good story", 1)[0]

                self.assertEqual(result["text"], self.data[4])

                # Test offsets still work after save/load
                self.embeddings.upsert([(0, "Looking out into the dreadful abyss", None)])
                self.assertEqual(self.embeddings.count(), len(self.data))

        def testAutoId(self):
            """
            Test auto id generation
            """

            # Default sequence id
            embeddings = Embeddings(path="sentence-transformers/nli-mpnet-base-v2", content=self.backend)
            embeddings.index(self.data)

            result = embeddings.search("feel good story", 1)[0]
            self.assertEqual(result["text"], self.data[4])

            # UUID
            embeddings.config["autoid"] = "uuid4"
            embeddings.index(self.data)

            result = embeddings.search(self.data[4], 1)[0]
            self.assertEqual(len(result["id"]), 36)

        def testCheckpoint(self):
            """
            Test embeddings index checkpoints
            """

            # Checkpoint directory
            checkpoint = os.path.join(tempfile.gettempdir(), f"embeddings.{self.category()}.checkpoint")

            # Save embeddings checkpoint
            self.embeddings.index(self.data, checkpoint=checkpoint)

            # Reindex with checkpoint
            self.embeddings.index(self.data, checkpoint=checkpoint)

            # Search for best match
            result = self.embeddings.search("feel good story", 1)[0]
            self.assertEqual(result["text"], self.data[4])

        def testColumns(self):
            """
            Test custom text/object columns
            """

            embeddings = Embeddings({"keyword": True, "content": self.backend, "columns": {"text": "value"}})
            data = [{"value": x} for x in self.data]
            embeddings.index([(uid, text, None) for uid, text in enumerate(data)])

            # Run search
            result = embeddings.search("lottery", 1)[0]
            self.assertEqual(result["text"], self.data[4])

        def testClose(self):
            """
            Test embeddings close
            """

            embeddings = None

            # Create index twice to test open/close and ensure resources are freed
            for _ in range(2):
                embeddings = Embeddings(
                    {"path": "sentence-transformers/nli-mpnet-base-v2", "scoring": {"method": "bm25", "terms": True}, "content": self.backend}
                )

                # Add record to index
                embeddings.index([(0, "Close test", None)])

                # Save index
                index = os.path.join(tempfile.gettempdir(), f"embeddings.{self.category()}.close")
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
            embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": self.backend})
            self.assertEqual(embeddings.search("test"), [])

            # Test index with no data
            embeddings.index([])
            self.assertIsNone(embeddings.ann)

            # Test upsert with no data
            embeddings.index([(0, "this is a test", None)])
            embeddings.upsert([])
            self.assertIsNotNone(embeddings.ann)

        def testEmptyString(self):
            """
            Test empty string indexing
            """

            # Test empty string
            self.embeddings.index([(0, "", None)])
            self.assertTrue(self.embeddings.search("test"))

            # Test empty string with dict
            self.embeddings.index([(0, {"text": ""}, None)])
            self.assertTrue(self.embeddings.search("test"))

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

        def testHybrid(self):
            """
            Test hybrid search
            """

            # Build data array
            data = [(uid, text, None) for uid, text in enumerate(self.data)]

            # Index data with sparse + dense vectors.
            embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "hybrid": True, "content": self.backend})
            embeddings.index(data)

            # Run search
            result = embeddings.search("feel good story", 1)[0]
            self.assertEqual(result["text"], data[4][1])

            # Generate temp file path
            index = os.path.join(tempfile.gettempdir(), f"embeddings.{self.category()}.hybrid")

            # Test load/save
            embeddings.save(index)
            embeddings.load(index)

            # Run search
            result = embeddings.search("feel good story", 1)[0]
            self.assertEqual(result["text"], data[4][1])

            # Index data with sparse + dense vectors and unnormalized scores.
            embeddings.config["scoring"]["normalize"] = False
            embeddings.index(data)

            # Run search
            result = embeddings.search("feel good story", 1)[0]
            self.assertEqual(result["text"], data[4][1])

            # Test upsert
            data[0] = (0, "Feel good story: baby panda born", None)
            embeddings.upsert([data[0]])

            result = embeddings.search("feel good story", 1)[0]
            self.assertEqual(result["text"], data[0][1])

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
                {
                    "path": "sentence-transformers/nli-mpnet-base-v2",
                    "content": self.backend,
                    "instructions": {"query": "query: ", "data": "passage: "},
                }
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

        def testKeyword(self):
            """
            Test keyword only (sparse) search
            """

            # Build data array
            data = [(uid, text, None) for uid, text in enumerate(self.data)]

            # Index data with sparse + dense vectors
            embeddings = Embeddings({"keyword": True, "content": self.backend})
            embeddings.index(data)

            # Run search
            result = embeddings.search("lottery ticket", 1)[0]
            self.assertEqual(result["text"], data[4][1])

            # Test count method
            self.assertEqual(embeddings.count(), len(data))

            # Generate temp file path
            index = os.path.join(tempfile.gettempdir(), f"embeddings.{self.category()}.keyword")

            # Test load/save
            embeddings.save(index)
            embeddings.load(index)

            # Run search
            result = embeddings.search("lottery ticket", 1)[0]
            self.assertEqual(result["text"], data[4][1])

            # Update data
            data[0] = (0, "Feel good story: baby panda born", None)
            embeddings.upsert([data[0]])

            # Search for best match
            result = embeddings.search("feel good story", 1)[0]
            self.assertEqual(result["text"], data[0][1])

        def testMultiData(self):
            """
            Test indexing with multiple data types (text, documents)
            """

            embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": self.backend, "batch": len(self.data)})

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
            index = os.path.join(tempfile.gettempdir(), f"embeddings.{self.category()}.insert")
            self.embeddings.save(index)

            # Modify index
            self.embeddings.upsert([(0, "Looking out into the dreadful abyss", None)])

            # Save to a different location
            indexupdate = os.path.join(tempfile.gettempdir(), f"embeddings.{self.category()}.update")
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

        def testNoIndex(self):
            """
            Test an embeddings instance with no available indexes
            """

            # Disable top-level indexing
            embeddings = Embeddings(
                {
                    "content": self.backend,
                    "defaults": False,
                }
            )
            embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

            with self.assertRaises(IndexNotFoundError):
                embeddings.search("select id, text, score from txtai where similar('feel good story')")

        def testNotImplemented(self):
            """
            Test exceptions for non-implemented methods
            """

            db = RDBMS({})

            self.assertRaises(NotImplementedError, db.connect, None)
            self.assertRaises(NotImplementedError, db.getcursor)
            self.assertRaises(NotImplementedError, db.jsonprefix)
            self.assertRaises(NotImplementedError, db.jsoncolumn, None)
            self.assertRaises(NotImplementedError, db.rows)
            self.assertRaises(NotImplementedError, db.addfunctions)

            db = Embedded({})
            self.assertRaises(NotImplementedError, db.copy, None)

        def testObject(self):
            """
            Test object field
            """

            # Encode object
            embeddings = Embeddings({"defaults": False, "content": self.backend, "objects": True})
            embeddings.index([{"object": "binary data".encode("utf-8")}])

            # Decode and test extracted object
            obj = embeddings.search("select object from txtai where id = 0")[0]["object"]
            self.assertEqual(str(obj.getvalue(), "utf-8"), "binary data")

        def testPickle(self):
            """
            Test pickle configuration
            """

            embeddings = Embeddings(
                {
                    "format": "pickle",
                    "path": "sentence-transformers/nli-mpnet-base-v2",
                    "content": self.backend,
                }
            )

            embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

            # Generate temp file path
            index = os.path.join(tempfile.gettempdir(), f"embeddings.{self.category()}.pickle")

            embeddings.save(index)

            # Check that config exists
            self.assertTrue(os.path.exists(os.path.join(index, "config")))

            # Check that index can be reloaded
            embeddings.load(index)
            self.assertEqual(embeddings.count(), 6)

        def testQuantize(self):
            """
            Test scalar quantization
            """

            # Index data with 1-bit scalar quantization
            embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "quantize": 1, "content": self.backend})
            embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

            # Search for best match
            result = self.embeddings.search("feel good story", 1)[0]
            self.assertEqual(result["text"], self.data[4])

        def testQueryModel(self):
            """
            Test index
            """

            embeddings = Embeddings(
                {"path": "sentence-transformers/nli-mpnet-base-v2", "content": self.backend, "query": {"path": "neuml/t5-small-txtsql"}}
            )

            # Create an index for the list of text
            embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

            # Search for best match
            result = embeddings.search("feel good story with win in text", 1)[0]

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
            self.embeddings.reindex({"path": "sentence-transformers/nli-mpnet-base-v2"})

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
            index = os.path.join(tempfile.gettempdir(), f"embeddings.{self.category()}")

            self.embeddings.save(index)
            self.embeddings.load(index)

            # Search for best match
            result = self.embeddings.search("feel good story", 1)[0]

            self.assertEqual(result["text"], self.data[4])

            # Test offsets still work after save/load
            self.embeddings.upsert([(0, "Looking out into the dreadful abyss", None)])
            self.assertEqual(self.embeddings.count(), len(self.data))

        def testSettings(self):
            """
            Test custom SQLite settings
            """

            # Index with write-ahead logging enabled
            embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": self.backend, "sqlite": {"wal": True}})

            # Create an index for the list of text
            embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

            # Search for best match
            result = embeddings.search("feel good story", 1)[0]

            self.assertEqual(result["text"], self.data[4])

        def testSQL(self):
            """
            Test running a SQL query
            """

            # Create an index for the list of text
            self.embeddings.index([(uid, {"text": text, "length": len(text), "attribute": f"ID{uid}"}, None) for uid, text in enumerate(self.data)])

            # Test similar
            result = self.embeddings.search(
                "select text, score from txtai where similar('feel good story') group by text, score having count(*) > 0 order by score desc", 1
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

            # Test column filtering
            result = self.embeddings.search("select text from txtai where attribute = 'ID4'", 1)[0]
            self.assertEqual(result["text"], self.data[4])

            # Test SQL parse error
            with self.assertRaises(SQLError):
                self.embeddings.search("select * from txtai where bad,query")

        def testSQLBind(self):
            """
            Test SQL statements with bind parameters
            """

            # Create an index for the list of text
            self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

            # Test similar clause bind parameters
            result = self.embeddings.search("select id, text, score from txtai where similar(:x)", parameters={"x": "feel good story"})[0]
            self.assertEqual(result["text"], self.data[4])

            # Test similar clause bind and non-bind parameters
            result = self.embeddings.search("select id, text, score from txtai where similar(:x, 0.5)", parameters={"x": "feel good story"})[0]
            self.assertEqual(result["text"], self.data[4])

            # Test where filtering with bind parameters
            result = self.embeddings.search("select * from txtai where text like :x", parameters={"x": "%iceberg%"})[0]
            self.assertEqual(result["text"], self.data[1])

        def testSubindex(self):
            """
            Test subindex
            """

            # Build data array
            data = [(uid, text, None) for uid, text in enumerate(self.data)]

            # Disable top-level indexing and create subindex
            embeddings = Embeddings(
                {"content": self.backend, "defaults": False, "indexes": {"index1": {"path": "sentence-transformers/nli-mpnet-base-v2"}}}
            )
            embeddings.index(data)

            # Test transform
            self.assertEqual(embeddings.transform("feel good story").shape, (768,))

            # Run search
            result = embeddings.search("feel good story", 1)[0]
            self.assertEqual(result["text"], data[4][1])

            # Run SQL search
            result = embeddings.search("select id, text, score from txtai where similar('feel good story', 10, 0.5)")[0]
            self.assertEqual(result["text"], data[4][1])

            # Test missing index
            with self.assertRaises(IndexNotFoundError):
                embeddings.search("select id, text, score from txtai where similar('feel good story', 'notindex')")

            # Generate temp file path
            index = os.path.join(tempfile.gettempdir(), f"embeddings.{self.category()}.subindex")

            # Test load/save
            embeddings.save(index)
            embeddings.load(index)

            # Run search
            result = embeddings.search("feel good story", 1)[0]
            self.assertEqual(result["text"], data[4][1])

            # Update data
            data[0] = (0, "Feel good story: baby panda born", None)
            embeddings.upsert([data[0]])

            # Search for best match
            result = embeddings.search("feel good story", 1)[0]
            self.assertEqual(result["text"], data[0][1])

            # Check missing text is set to id when top-level indexing is disabled
            embeddings.upsert([(embeddings.count(), {"content": "empty text"}, None)])
            result = embeddings.search(f"{embeddings.count() - 1}", 1)[0]
            self.assertEqual(result["text"], str(embeddings.count() - 1))

            # Close embeddings
            embeddings.close()

        def testSubindexEmpty(self):
            """
            Test loading an empty subindex
            """

            # Build data array
            data = [(uid, {"column1": text}, None) for uid, text in enumerate(self.data)]

            # Disable top-level indexing and create subindexes
            embeddings = Embeddings(
                {
                    "content": self.backend,
                    "defaults": False,
                    "indexes": {
                        "index1": {"path": "sentence-transformers/nli-mpnet-base-v2", "columns": {"text": "column1"}},
                        "index2": {"path": "sentence-transformers/nli-mpnet-base-v2", "columns": {"text": "column2"}},
                    },
                }
            )
            embeddings.index(data)

            # Generate temp file path
            index = os.path.join(tempfile.gettempdir(), f"embeddings.{self.category()}.subindexempty")

            # Save index
            embeddings.save(index)

            # Test exists
            self.assertTrue(embeddings.exists(index))

            # Load index
            embeddings.load(index)

            # Test search
            result = embeddings.search("feel good story", 1)[0]
            self.assertEqual(result["text"], data[4][1]["text"])

        def testTerms(self):
            """
            Test extracting keyword terms from query
            """

            result = self.embeddings.terms("select * from txtai where similar('keyword terms')")
            self.assertEqual(result, "keyword terms")

        def testTruncate(self):
            """
            Test dimensionality truncation
            """

            # Truncate vectors to a specified number of dimensions
            embeddings = Embeddings(
                {"path": "sentence-transformers/nli-mpnet-base-v2", "dimensionality": 750, "content": self.backend, "vectors": {"revision": "main"}}
            )
            embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

            # Search for best match
            result = self.embeddings.search("feel good story", 1)[0]
            self.assertEqual(result["text"], self.data[4])

        def testUpsert(self):
            """
            Test upsert
            """

            # Build data array
            data = [(uid, text, None) for uid, text in enumerate(self.data)]

            # Reset embeddings for test
            self.embeddings.ann = None
            self.embeddings.database = None

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
                self.embeddings.database = None

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

        def category(self):
            """
            Content backend category.

            Returns:
                category
            """

            return self.__class__.__name__.lower().replace("test", "")
