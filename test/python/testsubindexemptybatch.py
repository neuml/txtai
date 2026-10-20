"""
Subindex empty batch tests

These tests intentionally avoid sentence-transformers so they run without torch,
weights or downloaded test data.
"""

import unittest

from txtai import Embeddings


class TestSubindexEmptyBatch(unittest.TestCase):
    """
    Empty index and upsert batches must work with subindexes configured.
    """

    def testIndex(self):
        """
        Test empty index batch with a subindex
        """

        with Embeddings({"keyword": True, "indexes": {"terms": {"keyword": True}}}) as embeddings:
            embeddings.index([])
            self.assertEqual(embeddings.count(), 0)

    def testUpsert(self):
        """
        Test empty upsert batch with a subindex
        """

        with Embeddings({"keyword": True, "indexes": {"terms": {"keyword": True}}}) as embeddings:
            embeddings.upsert([])
            self.assertEqual(embeddings.count(), 0)

    def testRoundTrip(self):
        """
        Test empty batches before and after a non-empty batch
        """

        with Embeddings({"keyword": True, "indexes": {"terms": {"keyword": True}}}) as embeddings:
            # Empty batches on a fresh index
            embeddings.index([])
            embeddings.upsert([])
            self.assertEqual(embeddings.count(), 0)

            # A non-empty batch still indexes normally
            embeddings.upsert([(1, "alpha beta", None)])
            self.assertEqual(embeddings.count(), 1)

            # An empty upsert leaves existing content in place
            embeddings.upsert([])
            self.assertEqual(embeddings.count(), 1)

            # An empty index resets the index, matching the behavior without subindexes
            embeddings.index([])
            self.assertEqual(embeddings.count(), 0)

    def testWithoutSubindex(self):
        """
        Control: the same calls already worked without subindexes
        """

        with Embeddings({"keyword": True}) as embeddings:
            embeddings.index([])
            embeddings.upsert([])
            self.assertEqual(embeddings.count(), 0)
