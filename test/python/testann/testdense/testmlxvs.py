"""
mlxvs ANN module tests
"""

import importlib
import os
import sys
import tempfile
import types

from unittest.mock import patch

import numpy as np

from txtai.ann import ANNFactory

from .base import DenseTest


class FakeCagraIndex:
    """
    Stands in for mlxvs.CagraIndex in tests: exact NumPy search, honoring the filter.
    """

    def __init__(self, dataset, graph, metric):
        self.dataset, self.graph, self.metric = np.asarray(dataset, dtype=np.float32), np.asarray(graph), metric

    # pylint: disable=W0613,W0622
    def search(self, queries, k, itopk_size=64, filter=None):
        """
        Exact inner product search.
        """

        scores = queries @ self.dataset.T
        if filter is not None:
            scores[:, ~np.asarray(filter)] = -np.inf

        ids = np.argsort(-scores, axis=1)[:, :k]
        scores = np.take_along_axis(scores, ids, axis=1)
        return np.where(np.isinf(scores), -1, ids), scores


def fakecagrabuild(vectors, degree, metric, intermediate_graph_degree):
    """
    Stands in for mlxvs.cagra_build in tests.
    """

    # pylint: disable=W0613
    return np.zeros((vectors.shape[0], degree), dtype=np.int32)


def fakemodules():
    """
    Fake mlx.core and mlxvs modules, so the GPU code path runs (and is covered) on every platform.

    Returns:
        {module name: module}
    """

    core = types.ModuleType("mlx.core")
    core.array, core.float32, core.float16 = np.asarray, np.float32, np.float16

    mlx = types.ModuleType("mlx")
    mlx.core = core

    mlxvs = types.ModuleType("mlxvs")
    mlxvs.CagraIndex, mlxvs.cagra_build = FakeCagraIndex, fakecagrabuild

    return {"mlx": mlx, "mlx.core": core, "mlxvs": mlxvs}


class TestMlxvs(DenseTest):
    """
    mlxvs ANN tests. On Apple silicon with mlxvs installed they run the GPU graph search, elsewhere the exact NumPy
    fallback.
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

    @patch("txtai.ann.dense.mlxvs.MLXVS", False)
    def testMlxvsFallback(self):
        """
        Test the exact NumPy search used when mlxvs isn't available
        """

        self.runTests("mlxvs")

    def testMlxvsMocked(self):
        """
        Test the GPU code path with mlx and mlxvs mocked, so it runs on platforms without them
        """

        name = "txtai.ann.dense.mlxvs"
        with patch.dict(sys.modules, fakemodules()):
            importlib.reload(sys.modules[name])

            # Standard tests on the graph path
            self.runTests("mlxvs")

            rng = np.random.default_rng(0)
            data = rng.random((1000, 32), dtype=np.float32)
            self.normalize(data)

            model = ANNFactory.create({"backend": "mlxvs", "dimensions": 32})
            model.index(data)
            self.assertIsInstance(model.backend, FakeCagraIndex)

            # Deleted rows are filtered out of graph search results
            model.delete([0, 1])
            results = model.search(data[:2], 5)
            self.assertTrue(all(uid > 1 for result in results for uid, _ in result))

            # A limit beyond the graph search's list is searched exactly
            self.assertEqual(len(model.search(data[:1], 1500)[0]), 998)

            # Save and load keep the graph
            path = os.path.join(tempfile.gettempdir(), "ann.mlxvs.mocked")
            model.save(path)
            model.load(path)
            self.assertIsInstance(model.backend, FakeCagraIndex)
            self.assertEqual(model.count(), 998)

            # Every row deleted: no results
            model.delete(list(range(1000)))
            self.assertEqual(model.search(data[:2], 5), [[], []])

        # Reload the module as the platform imports it
        importlib.reload(sys.modules[name])

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
