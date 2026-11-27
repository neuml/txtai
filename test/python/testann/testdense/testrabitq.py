"""
RaBitQ ANN module tests
"""

import os
import tempfile
import time

import numpy as np

from rabitqlib import HnswIndex, IvfIndex

from txtai.ann import ANNFactory

from .base import DenseTest


class TestRaBitQ(DenseTest):
    """
    RaBitQ ANN tests.
    """

    def testRabitQ(self):
        """
        Test RaBitQ backend
        """

        self.runTests("rabitq")

    def testRabitQAppendDelete(self):
        """
        Test RaBitQ ivf appends and deletes rows without retaining vectors
        """

        # Generate dummy data
        data = np.random.rand(100, 240).astype(np.float32)
        self.normalize(data)

        # Raw vectors rank exactly, ivf probes every cluster
        config = {"backend": "rabitq", "dimensions": 240, "rabitq": {"nbits": 32, "nprobe": 100}}
        ann = ANNFactory.create(config)
        ann.index(data)

        # Appended rows continue the ids
        ann.append(data[:10])
        self.assertEqual(ann.count(), 110)
        self.assertEqual(ann.config["offset"], 110)
        self.assertEqual({uid for uid, _ in ann.search(data[5:6], 2)[0]}, {5, 105})

        # Unknown and repeated ids are ignored, deleted rows are skipped by search and the limit is still filled
        ann.delete([5, 5, 1000, -1])
        self.assertEqual(ann.count(), 109)
        for result in ann.search(data[:20], 10):
            self.assertEqual(len(result), 10)
            self.assertNotIn(5, {uid for uid, _ in result})

        # Deletes are kept in a new instance created from the saved config, like Embeddings.load
        index = os.path.join(tempfile.gettempdir(), f"rabitq.delete.{round(time.time() * 1000)}")
        ann.save(index)
        loaded = ANNFactory.create(dict(ann.config))
        loaded.load(index)
        self.assertEqual(loaded.count(), 109)
        self.assertEqual(loaded.search(data[:3], 5), ann.search(data[:3], 5))

        # A deleted row stays hidden after a reload, even for a query equal to its vector
        self.assertNotIn(5, {uid for uid, _ in loaded.search(data[5:6], 10)[0]})

        # Appends and deletes continue after a reload
        loaded.append(data[:5])
        self.assertEqual(loaded.count(), 114)
        loaded.delete([0])
        self.assertEqual(loaded.count(), 113)

        # Deleting every row leaves an empty index
        loaded.delete(list(range(200)))
        self.assertEqual(loaded.count(), 0)
        self.assertEqual(loaded.search(data[:1], 10), [[]])

    def testRabitQCustom(self):
        """
        Test RaBitQ backend with custom settings
        """

        # Test with custom settings, hnsw indexes don't support append and delete
        self.runTests("rabitq", {"rabitq": {"mode": "hnsw"}}, False)
        self.runTests("rabitq", {"rabitq": {"clusters": 8, "nprobe": 2}}, False)
        self.runTests("rabitq", {"rabitq": {"nbits": 4}}, False)
        self.runTests("rabitq", {"rabitq": {"mode": "hnsw", "nbits": 4}}, False)
        self.runTests("rabitq", {"rabitq": {"nbits": 32}})

        # Generate dummy data
        data = np.random.rand(100, 240).astype(np.float32)
        self.normalize(data)

        # Test invalid modes and quantization bits
        for mode, nbits in [("invalid", 1), ("ivf", 10), ("hnsw", 32)]:
            with self.assertRaises(ValueError):
                ann = ANNFactory.create({"backend": "rabitq", "dimensions": 240, "rabitq": {"mode": mode, "nbits": nbits}})
                ann.index(data)

        # Test a failed load leaves an empty index
        ann = ANNFactory.create({"backend": "rabitq", "dimensions": 240})
        ann.index(data)
        index = os.path.join(tempfile.gettempdir(), f"rabitq.invalid.{round(time.time() * 1000)}")
        ann.save(index)
        ann.config["rabitq"] = {"mode": "invalid"}
        with self.assertRaises(ValueError):
            ann.load(index)
        self.assertEqual(ann.count(), 0)

    def testRabitQNativeCounts(self):
        """
        Test native counts after loading deleted rows and spare HNSW capacity.
        """

        data = np.random.default_rng(0).random((8, 64), dtype=np.float32)
        self.normalize(data)
        centroids = data.mean(axis=0, keepdims=True)
        clusterids = np.zeros(len(data), dtype=np.uint32)

        for mode in ["ivf", "hnsw"]:
            if mode == "ivf":
                native = IvfIndex(64, 1, nbits=4, metric="ip")
            else:
                # Spare capacity beyond the stored rows
                native = HnswIndex(64, nbits=4, num_clusters=1, metric="ip")
                native.reserve(16)

            native.set_centroids(centroids)
            native.add(data, cluster_ids=clusterids)
            native.remove(np.array([0], dtype=np.int64))

            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "index")
                native.save(path)

                # Old saved configs may still have a deletes counter; native counts already exclude it.
                for metadata in [{}, {"deletes": 1}]:
                    with self.subTest(mode=mode, metadata=metadata):
                        config = {"backend": "rabitq", "dimensions": 64, "rabitq": {"mode": mode, "nprobe": 1}, **metadata}
                        ann = ANNFactory.create(config)
                        self.assertEqual(ann.count(), 0)
                        ann.load(path)
                        self.assertEqual(ann.count(), 7)
                        self.assertEqual(ann.backend.num_stored_points, 8)

                        if mode == "ivf":
                            # Deleted slots still occupy IDs; appended IDs must not reuse them.
                            ann.append(data[:1])
                            self.assertEqual(ann.config["offset"], 9)
                            self.assertEqual(ann.count(), 8)
                            ann.delete([8, 8, 9, -1])
                            self.assertEqual(ann.count(), 7)
                            ann.delete(list(range(9)))
                            self.assertEqual(ann.count(), 0)
                            self.assertEqual(ann.search(data[:1], 10), [[]])

    def testRabitQUnfilled(self):
        """
        Test unfilled result slots are dropped
        """

        # Generate dummy data
        data = np.random.rand(500, 240).astype(np.float32)
        self.normalize(data)

        # Probing one of ten clusters leaves part of each result list unfilled
        ann = ANNFactory.create({"backend": "rabitq", "dimensions": 240, "rabitq": {"clusters": 10, "nprobe": 1}})
        ann.index(data)

        results = ann.search(data[:5], data.shape[0])
        for result in results:
            self.assertLess(len(result), data.shape[0])
            self.assertTrue(all(uid >= 0 and np.isfinite(score) for uid, score in result))

    def testRabitQUpdate(self):
        """
        Test RaBitQ stores a single file and hnsw does not support append and delete
        """

        # Generate dummy data
        data = np.random.rand(100, 240).astype(np.float32)
        self.normalize(data)

        for mode in ["ivf", "hnsw"]:
            ann = ANNFactory.create({"backend": "rabitq", "dimensions": 240, "rabitq": {"mode": mode, "nprobe": 100}})
            ann.index(data)

            # Offset marks the index as existing, see Embeddings.exists
            self.assertEqual(ann.config["offset"], 100)

            if mode == "hnsw":
                with self.assertRaises(NotImplementedError):
                    ann.append(data[:1])

                with self.assertRaises(NotImplementedError):
                    ann.delete([0])

            # Limit above the row count returns every row
            for result in ann.search(data[:2], 200):
                self.assertEqual(len(result), 100)

            # Index is a single file that reloads with the same count and results
            index = os.path.join(tempfile.gettempdir(), f"rabitq.{mode}.{round(time.time() * 1000)}")
            ann.save(index)
            self.assertTrue(os.path.isfile(index))

            # File is the native index, readable without the wrapper
            native = IvfIndex.load(index) if mode == "ivf" else HnswIndex.load(index)
            self.assertEqual(native.num_stored_points, 100)

            expected = ann.search(data[:2], 10)
            ann.load(index)
            self.assertEqual(ann.count(), 100)
            self.assertEqual(ann.search(data[:2], 10), expected)
