"""
Reranker module tests
"""

import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch

from txtai import Embeddings
from txtai.pipeline import Reranker, Similarity


class TestReranker(unittest.TestCase):
    """
    Reranker tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create single labels instance.
        """

        cls.data = [
            "US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day",
        ]

        cls.embeddings = Embeddings(content=True)
        cls.embeddings.index(cls.data)
        cls.similarity = Similarity("neuml/colbert-bert-tiny", lateencode=True)

    def assertScores(self, cached, uncached):
        """Compare candidate order and scores at the supported precision."""

        self.assertEqual([row.get("id") for row in cached], [row.get("id") for row in uncached])
        self.assertEqual([row["text"] for row in cached], [row["text"] for row in uncached])
        for actual, expected in zip(cached, uncached):
            self.assertAlmostEqual(actual["score"], expected["score"], places=4)

    @staticmethod
    def dataCalls(encode):
        """Return only document encoding calls."""

        return [call for call in encode.call_args_list if call.args[1] == "data"]

    @staticmethod
    def lockRecorder(ranker, encode, calls):
        """Record encoder lock ownership using a separate thread."""

        def record(data, category):
            held = []

            def probe():
                acquired = ranker.lock.acquire(blocking=False)
                held.append(not acquired)
                if acquired:
                    ranker.lock.release()

            thread = threading.Thread(target=probe)
            thread.start()
            thread.join()
            calls.append((category, held[0]))
            return encode(data, category)

        return record

    def testCacheLock(self):
        """Hold the cache lock across document and query encodes while leaving default calls unlocked."""

        for enabled in (True, False):
            ranker = Reranker(self.embeddings, self.similarity, cache=True) if enabled else Reranker(self.embeddings, self.similarity)
            calls = []
            record = self.lockRecorder(ranker, self.similarity.lateencoder.encode, calls)
            with patch.object(self.similarity.lateencoder, "encode", side_effect=record):
                ranker("lottery")
                ranker("lottery")
            self.assertEqual({category for category, _ in calls}, {"data", "query"})
            self.assertTrue(all(held == enabled for _, held in calls), calls)

    def testCacheShared(self):
        """Serialize cached rerankers that share a similarity pipeline on one lock."""

        first = Reranker(self.embeddings, self.similarity, cache=True)
        second = Reranker(self.embeddings, self.similarity, cache=True)
        calls = []
        with patch.object(self.similarity.lateencoder, "encode", side_effect=self.lockRecorder(second, self.similarity.lateencoder.encode, calls)):
            first("lottery")
        self.assertEqual({category for category, _ in calls}, {"data", "query"})
        self.assertTrue(all(held for _, held in calls), calls)

    def testCacheArgument(self):
        """Coerce cache bounds to integers and disable non-positive bounds."""

        for cache, bound in ((True, 1000), (3, 3), ("2", 2), (2.5, 2)):
            ranker = Reranker(self.embeddings, self.similarity, cache=cache)
            self.assertEqual(ranker.cachesize, bound)
            self.assertIsNotNone(ranker.cache)
        for cache in (None, False, 0, -1):
            self.assertIsNone(Reranker(self.embeddings, self.similarity, cache=cache).cache)

    def testCacheHits(self):
        """Reuse document vectors on both the second and third rescoring."""

        ranker = Reranker(self.embeddings, self.similarity, cache=True)
        expected = Reranker(self.embeddings, self.similarity)("lottery winner")
        with patch.object(self.similarity.lateencoder, "encode", wraps=self.similarity.lateencoder.encode) as encode:
            self.assertScores(ranker("lottery winner"), expected)
            self.assertEqual(len(self.dataCalls(encode)), 1)
            for _ in range(2):
                encode.reset_mock()
                self.assertScores(ranker("lottery winner"), expected)
                self.assertEqual(len(self.dataCalls(encode)), 0)
        self.assertEqual(ranker.cachesize, 1000)

    def testCacheMixed(self):
        """Pad mixed hits and misses in candidate order for batched queries."""

        ranker = Reranker(self.embeddings, self.similarity, cache=True)
        ranker("select id, text, score from txtai where id = '4'")
        queries = ["select id, text, score from txtai order by id", "select id, text, score from txtai order by id desc"]
        expected = Reranker(self.embeddings, self.similarity)(queries)
        with patch.object(self.similarity.lateencoder, "encode", wraps=self.similarity.lateencoder.encode) as encode:
            actual = ranker(queries)
            self.assertEqual(len(self.dataCalls(encode)), 1)
            self.assertEqual(len(self.dataCalls(encode)[0].args[0]), 5)
            for cached, uncached in zip(actual, expected):
                self.assertScores(cached, uncached)
        self.assertGreater(len({value.shape[0] for value in ranker.cache.values()}), 1)
        for value in ranker.cache.values():
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(value.device.type, "cpu")
            self.assertTrue((value.abs().sum(dim=-1) > 0).all())

    def testCacheTextChanges(self):
        """Re-encode changed text while rebuilds and embeddings swaps reuse unchanged text."""

        embeddings = Embeddings(content=True)
        embeddings.index([(0, self.data[4], None)])
        ranker = Reranker(embeddings, self.similarity, cache=True)
        ranker("lottery")
        with patch.object(self.similarity.lateencoder, "encode", wraps=self.similarity.lateencoder.encode) as encode:
            embeddings.upsert([(0, self.data[0], None)])
            self.assertScores(ranker("lottery"), Reranker(embeddings, self.similarity)("lottery"))
            self.assertEqual(len(self.dataCalls(encode)), 2)
            embeddings.index([(0, self.data[0], None)])
            replacement = Embeddings(content=True)
            replacement.index([(0, self.data[0], None)])
            ranker.embeddings = replacement
            encode.reset_mock()
            ranker("lottery")
            self.assertEqual(len(self.dataCalls(encode)), 0)

    def testDisabled(self):
        """Keep the default string input path without building cache state."""

        ranker = Reranker(self.embeddings, self.similarity)
        with patch.object(self.similarity.lateencoder, "encode", wraps=self.similarity.lateencoder.encode) as encode:
            ranker("lottery")
            self.assertTrue(all(isinstance(text, str) for text in self.dataCalls(encode)[0].args[0]))
        self.assertIsNone(getattr(ranker, "cache", None))

    def testCacheSnapshot(self):
        """Restore cached document vectors and preserve typed ids and LRU order."""

        ranker = Reranker(self.embeddings, self.similarity, cache=True)
        expected = ranker("lottery")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cache.safetensors")
            ranker.save(path)
            restored = Reranker(self.embeddings, self.similarity, cache=True)
            restored.load(path)
            with patch.object(self.similarity.lateencoder, "encode", wraps=self.similarity.lateencoder.encode) as encode:
                self.assertScores(restored("lottery"), expected)
                self.assertEqual(len(self.dataCalls(encode)), 0)
            bounded = Reranker(self.embeddings, self.similarity, cache=2)
            bounded.load(path)
            self.assertEqual(list(bounded.cache), list(ranker.cache)[-2:])

            # An external integer id and its string spelling must remain distinct.
            rows = [[{"id": 1, "text": self.data[0]}, {"id": "1", "text": self.data[0]}]]
            typed = Reranker(self.embeddings, self.similarity, cache=True)
            with patch.object(self.embeddings, "batchsearch", return_value=rows):
                typed("lottery")
            typed.save(path)
            restored.load(path)
            self.assertEqual([key[1] for key in restored.cache.keys()], [1, "1"])

    def testCacheMismatch(self):
        """Discard mismatched snapshots with a warning and invalidate a replaced encoder."""

        ranker = Reranker(self.embeddings, self.similarity, cache=True)
        ranker("lottery")
        centered = Similarity("neuml/colbert-bert-tiny", lateencode=True, vectors={"center": {"scope": "document"}})
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cache.safetensors")
            ranker.save(path)
            restored = Reranker(self.embeddings, centered, cache=True)
            with self.assertLogs("txtai.pipeline.text.reranker", level="WARNING"):
                restored.load(path)
            self.assertFalse(restored.cache)
            with patch.object(centered.lateencoder, "encode", wraps=centered.lateencoder.encode) as encode:
                restored("lottery")
                self.assertEqual(len(self.dataCalls(encode)), 1)
                encode.reset_mock()
                ranker.similarity = centered
                self.assertScores(ranker("lottery"), restored("lottery"))
                self.assertEqual(len(self.dataCalls(encode)), 1)

    def testCacheMean(self):
        """Include a collection mean in the namespace and invalidate changed means."""

        mean = np.zeros(128, dtype=np.float32)
        similarity = Similarity("neuml/colbert-bert-tiny", lateencode=True, vectors={"center": {"scope": "collection", "mean": mean}})
        ranker = Reranker(self.embeddings, similarity, cache=True)
        self.assertScores(ranker("lottery"), Reranker(self.embeddings, similarity)("lottery"))
        namespace = ranker.namespace
        similarity.lateencoder.model.center["mean"] = np.ones(128, dtype=np.float32) / 100
        with patch.object(similarity.lateencoder, "encode", wraps=similarity.lateencoder.encode) as encode:
            ranker("lottery")
            self.assertEqual(len(self.dataCalls(encode)), 1)
        self.assertNotEqual(namespace, ranker.namespace)

    def testCacheEmpty(self):
        """Skip encoding and scoring when the cached candidate batch is empty."""

        ranker = Reranker(self.embeddings, self.similarity, cache=True)
        with patch.object(self.similarity.lateencoder, "encode", wraps=self.similarity.lateencoder.encode) as encode:
            self.assertEqual(ranker("select id, text, score from txtai where id = 'missing'"), [])
            self.assertEqual(encode.call_count, 0)

    def testEmpty(self):
        """Return no results when a search has no matches, with or without a cache."""

        empty = "select id, text, score from txtai where id = 'missing'"
        models = [
            self.similarity,
            Similarity("cross-encoder/ms-marco-MiniLM-L-2-v2", crossencode=True),
            Similarity("prajjwal1/bert-medium-mnli"),
        ]

        for similarity in models:
            for cache in (None, True):
                ranker = Reranker(self.embeddings, similarity, cache=cache)
                self.assertEqual(ranker(empty), [])

                # Empty and non-empty queries in the same batch
                results = ranker([empty, "lottery winner"], 1)
                self.assertEqual(results[0], [])
                self.assertEqual(len(results[1]), 1)

    def testCacheBypass(self):
        """Keep text scoring for missing ids, non-late encoders and batch centering."""

        query = "select text, score from txtai where similar('lottery')"
        ranker = Reranker(self.embeddings, self.similarity, cache=True)
        self.assertScores(ranker(query), Reranker(self.embeddings, self.similarity)(query))
        self.assertFalse(ranker.cache)
        for similarity in [
            Similarity("cross-encoder/ms-marco-MiniLM-L-2-v2", crossencode=True),
            Similarity("prajjwal1/bert-medium-mnli"),
            Similarity("neuml/colbert-bert-tiny", lateencode=True, vectors={"center": True}),
        ]:
            ranker = Reranker(self.embeddings, similarity, cache=True)
            self.assertScores(ranker("lottery"), Reranker(self.embeddings, similarity)("lottery"))
            self.assertFalse(ranker.cache)

    def testCacheBound(self):
        """Evict the least recently used document without losing current batch vectors."""

        ranker = Reranker(self.embeddings, self.similarity, cache=2)
        for uid in (0, 1, 0, 2):
            ranker(f"select id, text, score from txtai where id = '{uid}'")
            self.assertLessEqual(len(ranker.cache), 2)
        self.assertEqual([key[1] for key in ranker.cache.keys()], ["0", "2"])
        expected = Reranker(self.embeddings, self.similarity)("lottery")
        self.assertScores(ranker("lottery"), expected)
        self.assertEqual(len(ranker.cache), 2)

    def testCacheDisabledSnapshot(self):
        """Make disabled snapshots no-ops without accessing a file or an encoder."""

        ranker = Reranker(self.embeddings, self.similarity)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cache.safetensors")
            ranker.save(path)
            ranker.load(path)
            self.assertFalse(os.path.exists(path))

    def testCacheEmptySnapshot(self):
        """Round-trip an empty cache and bypass snapshots for unsupported encoders."""

        ranker = Reranker(self.embeddings, self.similarity, cache=True)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cache.safetensors")
            ranker.save(path)
            ranker.load(path)
            self.assertFalse(ranker.cache)
            ranker.similarity = Similarity("neuml/colbert-bert-tiny", lateencode=True, vectors={"center": True})
            other = os.path.join(directory, "unused.safetensors")
            ranker.save(other)
            ranker.load(other)
            self.assertFalse(os.path.exists(other))

    def testRanker(self):
        """
        Test re-ranking pipeline
        """

        embeddings = Embeddings(content=True)
        embeddings.index(self.data)

        similarity = Similarity("neuml/colbert-bert-tiny", lateencode=True)

        ranker = Reranker(embeddings, similarity)
        self.assertEqual(ranker("lottery winner")[0]["id"], "4")
