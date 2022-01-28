"""
Embeddings module tests
"""

import contextlib
import io
import os
import tempfile
import unittest

from unittest.mock import patch

import numpy as np

from txtai.embeddings import Embeddings
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

    def testExternal(self):
        """
        Test embeddings backed by external vectors
        """

        def transform(document):
            return document[1].astype(np.float32)

        # Generate random data and index
        data = np.random.rand(5, 100)
        embeddings = Embeddings({"method": "external", "transform": transform})
        embeddings.index([(x, row, None) for x, row in enumerate(data)])

        # Run search
        self.assertGreaterEqual(len(embeddings.search(data[0])), 1)

        # Run info
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            embeddings.info()

        self.assertTrue("txtai" in output.getvalue())

    def testIndex(self):
        """
        Test index
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Search for best match
        uid = self.embeddings.search("feel good story", 1)[0][0]

        self.assertEqual(uid, 4)

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
        cpucount.return_value = 2

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
