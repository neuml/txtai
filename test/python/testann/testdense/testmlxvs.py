"""
mlxvs ANN module tests
"""

import platform
import unittest

import numpy as np

from txtai.ann import ANNFactory

from .base import DenseTest


@unittest.skipIf(platform.system() != "Darwin" or platform.machine() != "arm64", "mlxvs requires macOS on Apple silicon")
class TestMlxvs(DenseTest):
    """
    mlxvs ANN tests.
    """

    def testMlxvs(self):
        """
        Test mlxvs backend
        """

        self.runTests("mlxvs")

    def testMlxvsCustom(self):
        """
        Test mlxvs backend with custom settings
        """

        self.runTests("mlxvs", {"mlxvs": {"graphdegree": 16, "intermediategraphdegree": 32, "itopk": 128, "dtype": "float16"}})

    def testMlxvsDeleteNumPy(self):
        """
        Test deleting ids passed as a NumPy array
        """

        self.deletenumpy("mlxvs")

    def testMlxvsRecall(self):
        """
        Test that graph search finds the exact nearest neighbors, and that deleted rows are not returned
        """

        rng = np.random.default_rng(0)
        data = rng.standard_normal((5000, 64)).astype(np.float32)
        self.normalize(data)

        model = ANNFactory.create({"backend": "mlxvs", "dimensions": 64})
        model.index(data)

        queries = data[:100]
        results = model.search(queries, 10)
        exact = np.argsort(-(queries @ data.T), axis=1)[:, :10]
        recall = np.mean([len({uid for uid, _ in result} & set(row)) / 10 for result, row in zip(results, exact)])
        self.assertGreater(recall, 0.95)

        # Each query is its own nearest neighbor until it is deleted
        model.delete(list(range(50)))
        results = model.search(queries, 10)
        self.assertTrue(all(uid >= 50 for result in results for uid, _ in result))
        self.assertTrue(all(result[0][0] == x for x, result in enumerate(results[50:], 50)))
