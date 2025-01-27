"""
Embeddings module tests
"""

import json
import os
import tempfile
import unittest

from unittest.mock import patch

import numpy as np

from txtai.embeddings import Embeddings, Reducer
from txtai.serialize import SerializeFactory


# pylint: disable=R0904
class TestEmbeddings(unittest.TestCase):
    """
    Embeddings tests.
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
        cls.embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})

    @classmethod
    def tearDownClass(cls):
        """
        Cleanup data.
        """

        if cls.embeddings:
            cls.embeddings.close()

    def testAutoId(self):
        """
        Test auto id generation
        """

        # Default sequence id
        embeddings = Embeddings()
        embeddings.index(self.data)

        uid = embeddings.search(self.data[4], 1)[0][0]
        self.assertEqual(uid, 4)

        # UUID
        embeddings = Embeddings(autoid="uuid4")
        embeddings.index(self.data)

        uid = embeddings.search(self.data[4], 1)[0][0]
        self.assertEqual(len(uid), 36)

    def testColumns(self):
        """
        Test custom text/object columns
        """

        embeddings = Embeddings({"keyword": True, "columns": {"text": "value"}})
        data = [{"value": x} for x in self.data]
        embeddings.index([(uid, text, None) for uid, text in enumerate(data)])

        # Run search
        uid = embeddings.search("lottery", 1)[0][0]
        self.assertEqual(uid, 4)

    def testContext(self):
        """
        Test embeddings context manager
        """

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "embeddings.context")

        with Embeddings() as embeddings:
            embeddings.index(self.data)
            embeddings.save(index)

        with Embeddings().load(index) as embeddings:
            uid = embeddings.search(self.data[4], 1)[0][0]
            self.assertEqual(uid, 4)

    def testDefaults(self):
        """
        Test default configuration
        """

        # Run index with no config which will fall back to default configuration
        embeddings = Embeddings()
        embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        self.assertEqual(embeddings.count(), 6)

    def testDelete(self):
        """
        Test delete
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Delete best match
        self.embeddings.delete([4])

        # Search for best match
        uid = self.embeddings.search("feel good story", 1)[0][0]

        self.assertEqual(self.embeddings.count(), 5)
        self.assertEqual(uid, 5)

    def testEmpty(self):
        """
        Test empty index
        """

        # Test search against empty index
        embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})
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

    def testExternal(self):
        """
        Test embeddings backed by external vectors
        """

        def transform(data):
            embeddings = []
            for text in data:
                # Create dummy embedding using sum and mean of character ordinals
                ordinals = [ord(c) for c in text]
                embeddings.append(np.array([sum(ordinals), np.mean(ordinals)]))

            return embeddings

        # Index data using simple embeddings transform method
        embeddings = Embeddings({"method": "external", "transform": transform})
        embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Run search
        uid = embeddings.search(self.data[4], 1)[0][0]
        self.assertEqual(uid, 4)

    def testExternalPrecomputed(self):
        """
        Test embeddings backed by external pre-computed vectors
        """

        # Test with no transform function
        data = np.random.rand(5, 10).astype(np.float32)

        embeddings = Embeddings({"method": "external"})
        embeddings.index([(uid, row, None) for uid, row in enumerate(data)])

        # Run search
        uid = embeddings.search(data[4], 1)[0][0]
        self.assertEqual(uid, 4)

    def testHybrid(self):
        """
        Test hybrid search
        """

        # Build data array
        data = [(uid, text, None) for uid, text in enumerate(self.data)]

        # Index data with sparse + dense vectors
        embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "hybrid": True})
        embeddings.index(data)

        # Run search
        uid = embeddings.search("feel good story", 1)[0][0]
        self.assertEqual(uid, 4)

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "embeddings.hybrid")

        # Test load/save
        embeddings.save(index)
        embeddings.load(index)

        # Run search
        uid = embeddings.search("feel good story", 1)[0][0]
        self.assertEqual(uid, 4)

        # Index data with sparse + dense vectors and unnormalized scores
        embeddings.config["scoring"]["normalize"] = False
        embeddings.index(data)

        # Run search
        uid = embeddings.search("feel good story", 1)[0][0]
        self.assertEqual(uid, 4)

        # Test upsert
        data[0] = (0, "Feel good story: baby panda born", None)
        embeddings.upsert([data[0]])

        uid = embeddings.search("feel good story", 1)[0][0]
        self.assertEqual(uid, 0)

    def testIds(self):
        """
        Test legacy config ids loading
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "embeddings.ids")

        # Save index
        self.embeddings.save(index)

        # Set ids on config to simulate legacy ids format
        with open(f"{index}/config.json", "r", encoding="utf-8") as handle:
            config = json.load(handle)
            config["ids"] = list(range(len(self.data)))

        with open(f"{index}/config.json", "w", encoding="utf-8") as handle:
            json.dump(config, handle, default=str, indent=2)

        # Reload index
        self.embeddings.load(index)

        # Run search
        uid = self.embeddings.search("feel good story", 1)[0][0]
        self.assertEqual(uid, 4)

        # Check that ids is not in config
        self.assertTrue("ids" not in self.embeddings.config)

    def testIdsPickle(self):
        """
        Test legacy pickle ids
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "embeddings.idspickle")

        # Save index
        self.embeddings.save(index)

        # Create ids as pickle
        path = os.path.join(tempfile.gettempdir(), "embeddings.idspickle", "ids")
        serializer = SerializeFactory.create("pickle", allowpickle=True)
        serializer.save(self.embeddings.ids.ids, path)

        # Reload index
        with self.assertWarns(FutureWarning):
            self.embeddings.load(index)

        # Run search
        uid = self.embeddings.search("feel good story", 1)[0][0]
        self.assertEqual(uid, 4)

    def testIndex(self):
        """
        Test index
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Search for best match
        uid = self.embeddings.search("feel good story", 1)[0][0]

        self.assertEqual(uid, 4)

    def testKeyword(self):
        """
        Test keyword only (sparse) search
        """

        # Build data array
        data = [(uid, text, None) for uid, text in enumerate(self.data)]

        # Index data with sparse + dense vectors
        embeddings = Embeddings({"keyword": True})
        embeddings.index(data)

        # Run search
        uid = embeddings.search("lottery ticket", 1)[0][0]
        self.assertEqual(uid, 4)

        # Test count method
        self.assertEqual(embeddings.count(), len(data))

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "embeddings.keyword")

        # Test load/save
        embeddings.save(index)
        embeddings.load(index)

        # Run search
        uid = embeddings.search("lottery ticket", 1)[0][0]
        self.assertEqual(uid, 4)

        # Update data
        data[0] = (0, "Feel good story: baby panda born", None)
        embeddings.upsert([data[0]])

        # Search for best match
        uid = embeddings.search("feel good story", 1)[0][0]
        self.assertEqual(uid, 0)

    def testQuantize(self):
        """
        Test scalar quantization
        """

        for ann in ["faiss", "numpy", "torch"]:
            # Index data with 1-bit scalar quantization
            embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "quantize": 1, "backend": ann})
            embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

            # Search for best match
            uid = embeddings.search("feel good story", 1)[0][0]
            self.assertEqual(uid, 4)

    def testReducer(self):
        """
        Test reducer model
        """

        # Test model with single PCA component
        data = np.random.rand(5, 5).astype(np.float32)
        reducer = Reducer(data, 1)

        # Generate query and keep original data to ensure it changes
        query = np.random.rand(1, 5).astype(np.float32)
        original = query.copy()

        # Run test
        reducer(query)
        self.assertFalse(np.array_equal(query, original))

        # Test model with multiple PCA components
        reducer = Reducer(data, 3)

        # Generate query and keep original data to ensure it changes
        query = np.random.rand(5).astype(np.float32)
        original = query.copy()

        # Run test
        reducer(query)
        self.assertFalse(np.array_equal(query, original))

    def testReducerLegacy(self):
        """
        Test reducer model with legacy model format
        """

        # Test model with single PCA component
        data = np.random.rand(5, 5).astype(np.float32)
        reducer = Reducer(data, 1)

        # Save legacy format
        path = os.path.join(tempfile.gettempdir(), "reducer")
        serializer = SerializeFactory.create("pickle", allowpickle=True)
        serializer.save(reducer.model, path)

        # Load legacy format
        reducer = Reducer()
        reducer.load(path)

        # Generate query and keep original data to ensure it changes
        query = np.random.rand(1, 5).astype(np.float32)
        original = query.copy()

        # Run test
        reducer(query)
        self.assertFalse(np.array_equal(query, original))

    def testSave(self):
        """
        Test save
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "embeddings.base")

        self.embeddings.save(index)
        self.embeddings.load(index)

        # Search for best match
        uid = self.embeddings.search("feel good story", 1)[0][0]

        self.assertEqual(uid, 4)

        # Test offsets still work after save/load
        self.embeddings.upsert([(0, "Looking out into the dreadful abyss", None)])
        self.assertEqual(self.embeddings.count(), len(self.data))

    def testSimilarity(self):
        """
        Test similarity
        """

        # Get best matching id
        uid = self.embeddings.similarity("feel good story", self.data)[0][0]

        self.assertEqual(uid, 4)

    def testSubindex(self):
        """
        Test subindex
        """

        # Build data array
        data = [(uid, text, None) for uid, text in enumerate(self.data)]

        # Disable top-level indexing and create subindex
        embeddings = Embeddings({"defaults": False, "indexes": {"index1": {"path": "sentence-transformers/nli-mpnet-base-v2"}}})
        embeddings.index(data)

        # Test transform
        self.assertEqual(embeddings.transform("feel good story").shape, (768,))
        self.assertEqual(embeddings.transform("feel good story", index="index1").shape, (768,))
        with self.assertRaises(KeyError):
            embeddings.transform("feel good story", index="index2")

        # Run search
        uid = embeddings.search("feel good story", 1)[0][0]
        self.assertEqual(uid, 4)

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "embeddings.subindex")

        # Test load/save
        embeddings.save(index)
        embeddings.load(index)

        # Run search
        uid = embeddings.search("feel good story", 1)[0][0]
        self.assertEqual(uid, 4)

        # Update data
        data[0] = (0, "Feel good story: baby panda born", None)
        embeddings.upsert([data[0]])

        # Search for best match
        uid = embeddings.search("feel good story", 10)[0][0]
        self.assertEqual(uid, 0)

        # Check missing text is set to id when top-level indexing is disabled
        embeddings.upsert([(embeddings.count(), {"content": "empty text"}, None)])
        uid = embeddings.search(f"{embeddings.count() - 1}", 1)[0][0]
        self.assertEqual(uid, embeddings.count() - 1)

        # Close embeddings
        embeddings.close()

    def testTruncate(self):
        """
        Test dimensionality truncation
        """

        # Truncate vectors to a specified number of dimensions
        embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "dimensionality": 750, "vectors": {"revision": "main"}})
        embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Search for best match
        uid = embeddings.search("feel good story", 1)[0][0]
        self.assertEqual(uid, 4)

    def testUpsert(self):
        """
        Test upsert
        """

        # Build data array
        data = [(uid, text, None) for uid, text in enumerate(self.data)]

        # Reset embeddings for test
        self.embeddings.ann = None
        self.embeddings.ids = None

        # Create an index for the list of text
        self.embeddings.upsert(data)

        # Update data
        data[0] = (0, "Feel good story: baby panda born", None)
        self.embeddings.upsert([data[0]])

        # Search for best match
        uid = self.embeddings.search("feel good story", 1)[0][0]

        self.assertEqual(uid, 0)

    @patch("os.cpu_count")
    def testWords(self, cpucount):
        """
        Test embeddings backed by word vectors
        """

        # Mock CPU count
        cpucount.return_value = 1

        # Create dataset
        data = [(x, row.split(), None) for x, row in enumerate(self.data)]

        # Create embeddings model, backed by word vectors
        embeddings = Embeddings({"path": "neuml/glove-6B-quantized", "scoring": "bm25", "pca": 3, "quantize": True})

        # Call scoring and index methods
        embeddings.score(data)
        embeddings.index(data)

        # Test search
        self.assertIsNotNone(embeddings.search("win", 1))

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "embeddings.wordvectors")

        # Test save/load
        embeddings.save(index)
        embeddings.load(index)

        # Test search
        self.assertIsNotNone(embeddings.search("win", 1))

    @patch("os.cpu_count")
    def testWordsUpsert(self, cpucount):
        """
        Test embeddings backed by word vectors with upserts
        """

        # Mock CPU count
        cpucount.return_value = 1

        # Create dataset
        data = [(x, row.split(), None) for x, row in enumerate(self.data)]

        # Create embeddings model, backed by word vectors
        embeddings = Embeddings({"path": "neuml/glove-6B/model.sqlite", "scoring": "bm25", "pca": 3})

        # Call scoring and index methods
        embeddings.score(data)
        embeddings.index(data)

        # Now upsert and override record
        data = [(0, "win win", None)]

        # Update scoring and run upsert
        embeddings.score(data)
        embeddings.upsert(data)

        # Test search after upsert
        uid = embeddings.search("win", 1)[0][0]
        self.assertEqual(uid, 0)
