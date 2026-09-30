"""
HNSW ANN module tests
"""

import os
import tempfile

import numpy as np

from txtai.ann import ANNFactory

from .base import DenseTest


class TestHNSW(DenseTest):
    """
    HNSW ANN tests.
    """

    def testHnsw(self):
        """
        Test Hnswlib backend
        """

        self.runTests("hnsw")

    def testHnswCustom(self):
        """
        Test Hnswlib backend with custom settings
        """

        # Test with custom settings
        self.runTests("hnsw", {"hnsw": {"efconstruction": 100, "m": 8, "randomseed": 0, "efsearch": 15}})

    def testHnswLoad(self):
        """
        Test Hnswlib backend with a configuration that doesn't have a delete counter
        """

        # Save an index
        index = os.path.join(tempfile.gettempdir(), "hnsw.load")
        self.backend("hnsw", length=100).save(index)

        # Open the index with a configuration that only has the settings needed to load it
        ann = ANNFactory.create({"backend": "hnsw", "dimensions": 240, "metric": "ip"})
        ann.load(index)

        # Validate count, search and delete
        self.assertEqual(ann.count(), 100)
        self.assertEqual(len(ann.search(np.random.rand(1, 240).astype(np.float32), 1)[0]), 1)

        ann.delete([0])
        self.assertEqual(ann.count(), 99)
