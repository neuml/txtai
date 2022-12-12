"""
Embeddings module tests
"""

import os
import tempfile
import unittest

from unittest.mock import patch

import numpy as np

from txtai.embeddings import Embeddings, Reducer
from txtai.vectors import WordVectors


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

    def testIndex(self):
        """
        Test index
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Search for best match
        uid = self.embeddings.search("feel good story", 1)[0][0]

        self.assertEqual(uid, 4)

    def testNormalize(self):
        """
        Test batch normalize and single input normalize are equal
        """

        # Generate data
        data1 = np.random.rand(5, 5).astype(np.float32)
        data2 = data1.copy()

        # Keep original data to ensure it changed
        original = data1.copy()

        # Normalize data
        self.embeddings.normalize(data1)
        for x in data2:
            self.embeddings.normalize(x)

        # Test both data arrays are the same and changed from original
        self.assertTrue(np.allclose(data1, data2))
        self.assertFalse(np.allclose(data1, original))

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
        uid = self.embeddings.search("feel good story", 1)[0][0]

        self.assertEqual(uid, 0)

    @patch("os.cpu_count")
    def testWords(self, cpucount):
        """
        Test embeddings backed by word vectors
        """

        # Mock CPU count
        cpucount.return_value = 1

        # Initialize model path
        path = os.path.join(tempfile.gettempdir(), "model")
        os.makedirs(path, exist_ok=True)

        # Build tokens file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as output:
            tokens = output.name
            for x in self.data:
                output.write(x + "\n")

        # Word vectors path
        vectors = os.path.join(path, "test-10d")

        # Build word vectors, if they don't already exist
        WordVectors.build(tokens, 10, 1, vectors)

        # Create dataset
        data = [(x, row.split(), None) for x, row in enumerate(self.data)]

        # Create embeddings model, backed by word vectors
        embeddings = Embeddings({"path": vectors + ".magnitude", "storevectors": True, "scoring": "bm25", "pca": 3, "quantize": True})

        # Call scoring and index methods
        embeddings.score(data)
        embeddings.index(data)

        # Test search
        self.assertIsNotNone(embeddings.search("win", 1))

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "wembeddings")

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

        # Initialize model path
        path = os.path.join(tempfile.gettempdir(), "model")

        # Word vectors path
        vectors = os.path.join(path, "test-10d")

        # Create dataset
        data = [(x, row.split(), None) for x, row in enumerate(self.data)]

        # Create embeddings model, backed by word vectors
        embeddings = Embeddings({"path": vectors + ".magnitude", "storevectors": True, "scoring": "bm25", "pca": 3})

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
