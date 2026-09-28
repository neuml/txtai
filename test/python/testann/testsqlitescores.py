"""
SQLite ANN score tests.
"""

import os
import platform
import tempfile
import unittest

import numpy as np

from txtai import Embeddings
from txtai.ann import ANNFactory


@unittest.skipIf(platform.system() == "Darwin", "SQLite extensions not supported on macOS")
class TestSQLiteScores(unittest.TestCase):
    """
    Verify SQLite similarity scores with real sqlite-vec queries.
    """

    def testBinary(self):
        """
        Binary scores are the fraction of matching bits, including fractional values.
        """

        for dimensions in (8, 16, 24):
            with self.subTest(dimensions=dimensions):
                data = self.vectors(dimensions)
                model = ANNFactory.create({"backend": "sqlite", "dimensions": dimensions, "sqlite": {"quantize": 1}})
                try:
                    model.index(data)
                    queries = data[[0, 3]]
                    results = model.search(queries, len(data))
                    for query, result in zip(queries, results):
                        expected = {i: float(np.mean((query > 0) == (row > 0))) for i, row in enumerate(data)}
                        self.assertEqual({i for i, _ in result}, set(expected))
                        for i, score in result:
                            self.assertAlmostEqual(score, expected[i], places=12)
                        scores = [score for _, score in result]
                        self.assertEqual(scores, sorted(scores, reverse=True))
                    self.assertEqual(len(model.search(queries[:1], 2)[0]), 2)
                finally:
                    model.close()

    def testCosine(self):
        """
        FLOAT32 and INT8 modes continue to return cosine similarity.
        """

        for quantize in (None, False, True, 8):
            with self.subTest(quantize=quantize):
                data = self.vectors(8)
                model = ANNFactory.create({"backend": "sqlite", "dimensions": 8, "sqlite": {"quantize": quantize}})
                try:
                    model.index(data)
                    result = dict(model.search(data[:1], len(data))[0])
                    for i, score in enumerate((1.0, 0.75, 0.0, -1.0)):
                        self.assertAlmostEqual(result[i], score, delta=0.02 if quantize else 1e-6)
                finally:
                    model.close()

    def testPersistence(self):
        """
        Scores survive save/load, appends and deletes with a custom table.
        """

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "vectors.sqlite")
            data = self.vectors(16)
            config = {"backend": "sqlite", "dimensions": 16, "sqlite": {"quantize": 1, "table": "custom_vectors"}}
            model = ANNFactory.create(config)
            try:
                model.index(data[:2])
                model.save(path)
                model.close()
                model.load(path)
                model.append(data[2:])
                model.delete([0])
                model.save(path)
                model.close()
                model.load(path)
                self.assertEqual(model.count(), 3)
                self.assertEqual(model.search(data[:1], 4)[0], [(1, 0.9375), (2, 0.5), (3, 0.0)])
            finally:
                model.close()

    def testEmbeddings(self):
        """
        Public search keeps nonidentical binary neighbors with positive similarity.
        """

        data = self.vectors(8)
        with Embeddings({"method": "external", "backend": "sqlite", "sqlite": {"quantize": 1}}) as embeddings:
            embeddings.index([(str(i), row, None) for i, row in enumerate(data)])
            self.assertEqual(embeddings.search(data[0], 4), [("0", 1.0), ("1", 0.875), ("2", 0.5)])

    def vectors(self, dimensions):
        """
        Build normalized vectors differing in zero, one, half or all sign bits.

        Args:
            dimensions: number of vector dimensions

        Returns:
            normalized float32 vectors
        """

        data = np.ones((4, dimensions), dtype=np.float32)
        data[1, 0] = -1
        data[2, : dimensions // 2] = -1
        data[3, :] = -1
        return data / np.sqrt(np.float32(dimensions))
