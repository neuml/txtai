"""
Embeddings module tests
"""

import os
import tempfile
import unittest

import numpy as np

from txtai.embeddings import Embeddings
from txtai.vectors import WordVectors

class TestEmbeddings(unittest.TestCase):
    """
    Embeddings tests
    """

    def setUp(self):
        """
        Initialize test data.
        """

        self.data = ["US tops 5 million confirmed virus cases",
                     "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
                     "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
                     "The National Park Service warns against sacrificing slower friends in a bear attack",
                     "Maine man wins $1M from $25 lottery ticket",
                     "Make huge profits without work, earn up to $100,000 a day"]

        # Create embeddings model, backed by sentence-transformers & transformers
        self.embeddings = Embeddings({"method": "transformers",
                                      "path": "sentence-transformers/bert-base-nli-mean-tokens"})

    def testIndex(self):
        """
        Test embeddings.index
        """

        # Create an index for the list of sections
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Search for best match
        uid = self.embeddings.search("feel good story", 1)[0][0]

        self.assertEqual(self.data[uid], self.data[4])

    def testSave(self):
        """
        Test embeddings.save
        """

        # Create an index for the list of sections
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "embeddings")

        self.embeddings.save(index)
        self.embeddings.load(index)

        # Search for best match
        uid = self.embeddings.search("feel good story", 1)[0][0]

        self.assertEqual(self.data[uid], self.data[4])

    def testSimilarity(self):
        """
        Test embeddings.similarity
        """

        # Get best matching id
        uid = np.argmax(self.embeddings.similarity("feel good story", self.data))

        self.assertEqual(self.data[uid], self.data[4])

    def testWords(self):
        """
        Test embeddings backed by word vectors
        """

        # Initialize model path
        path = os.path.join(tempfile.gettempdir(), "model")
        os.makedirs(path, exist_ok=True)

        # Build tokens file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as output:
            tokens = output.name
            for x in self.data:
                output.write(x + "\n")

        # Word vectors path
        vectors = os.path.join(path, "test-300d")

        # Build word vectors, if they don't already exist
        WordVectors.build(tokens, 300, 1, vectors)

        # Create dataset
        data = [(x, row, None) for x, row in enumerate(self.data)]

        # Create embeddings model, backed by word vectors
        embeddings = Embeddings({"path": vectors + ".magnitude",
                                 "storevectors": True,
                                 "scoring": "bm25",
                                 "pca": 3,
                                 "quantize": True})

        # Call scoring and index methods
        embeddings.score(data)
        embeddings.index(data)

        # Test search
        self.assertIsNotNone(embeddings.search("win", 1))
