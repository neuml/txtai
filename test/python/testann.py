"""
ANN module tests
"""

import os
import platform
import sys
import tempfile
import unittest

from unittest.mock import patch

import numpy as np

from txtai.ann import ANNFactory, ANN
from txtai.serialize import SerializeFactory


# pylint: disable=R0904
class TestANN(unittest.TestCase):
    """
    ANN tests.
    """

    def testAnnoy(self):
        """
        Test Annoy backend
        """

        self.runTests("annoy", None, False)

    def testAnnoyCustom(self):
        """
        Test Annoy backend with custom settings
        """

        # Test with custom settings
        self.runTests("annoy", {"annoy": {"ntrees": 2, "searchk": 1}}, False)

    def testCustomBackend(self):
        """
        Test resolving a custom backend
        """

        self.runTests("txtai.ann.Faiss")

    def testCustomBackendNotFound(self):
        """
        Test resolving an unresolvable backend
        """

        with self.assertRaises(ImportError):
            ANNFactory.create({"backend": "notfound.ann"})

    def testFaiss(self):
        """
        Test Faiss backend
        """

        self.runTests("faiss")

    def testFaissCustom(self):
        """
        Test Faiss backend with custom settings
        """

        # Test with custom settings
        self.runTests("faiss", {"faiss": {"nprobe": 2, "components": "PCA16,IDMap,SQ8", "sample": 1.0}}, False)
        self.runTests("faiss", {"faiss": {"components": "IVF,SQ8"}}, False)

    @patch("platform.system")
    def testFaissMacOS(self, system):
        """
        Test Faiss backend with macOS
        """

        # Run test
        system.return_value = "Darwin"

        # pylint: disable=C0415, W0611
        # Force reload of class
        name = "txtai.ann.faiss"
        module = sys.modules[name]
        del sys.modules[name]
        import txtai.ann.faiss

        # Run tests
        self.runTests("faiss")

        # Restore original module
        sys.modules[name] = module

    @unittest.skipIf(os.name == "nt", "mmap not supported on Windows")
    def testFaissMmap(self):
        """
        Test Faiss backend with mmap enabled
        """

        # Test to with mmap enabled
        self.runTests("faiss", {"faiss": {"mmap": True}}, False)

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
        self.runTests("hnsw", {"hnsw": {"efconstruction": 100, "m": 4, "randomseed": 0, "efsearch": 5}})

    def testNotImplemented(self):
        """
        Test exceptions for non-implemented methods
        """

        ann = ANN({})

        self.assertRaises(NotImplementedError, ann.load, None)
        self.assertRaises(NotImplementedError, ann.index, None)
        self.assertRaises(NotImplementedError, ann.append, None)
        self.assertRaises(NotImplementedError, ann.delete, None)
        self.assertRaises(NotImplementedError, ann.search, None, None)
        self.assertRaises(NotImplementedError, ann.count)
        self.assertRaises(NotImplementedError, ann.save, None)

    def testNumPy(self):
        """
        Test NumPy backend
        """

        self.runTests("numpy")

    @patch.dict(os.environ, {"ALLOW_PICKLE": "True"})
    def testNumPyLegacy(self):
        """
        Test NumPy backend with legacy pickled data
        """

        serializer = SerializeFactory.create("pickle", allowpickle=True)

        # Create output directory
        output = os.path.join(tempfile.gettempdir(), "ann.npy")
        path = os.path.join(output, "embeddings")
        os.makedirs(output, exist_ok=True)

        # Generate data and save as pickle
        data = np.random.rand(100, 240).astype(np.float32)
        serializer.save(data, path)

        ann = ANNFactory.create({"backend": "numpy"})
        ann.load(path)

        # Validate count
        self.assertEqual(ann.count(), 100)

    @patch("sqlalchemy.orm.Query.limit")
    def testPGVector(self, query):
        """
        Test PGVector backend
        """

        # Generate test record
        data = np.random.rand(1, 240).astype(np.float32)

        # Mock database query
        query.return_value = [(x, -1.0) for x in range(data.shape[0])]

        configs = [
            ("full", {"dimensions": 240}, {}, data),
            ("half", {"dimensions": 240}, {"precision": "half"}, data),
            ("binary", {"quantize": 1, "dimensions": 240 * 8}, {}, data.astype(np.uint8)),
        ]

        # Create ANN
        for name, config, pgvector, data in configs:
            path = os.path.join(tempfile.gettempdir(), f"pgvector.{name}.sqlite")
            ann = ANNFactory.create(
                {**{"backend": "pgvector", "pgvector": {**{"url": f"sqlite:///{path}", "schema": "txtai"}, **pgvector}}, **config}
            )

            # Test indexing
            ann.index(data)
            ann.append(data)

            # Validate search results
            self.assertEqual(ann.search(data, 1), [[(0, 1.0)]])

            # Validate save/load/delete
            ann.save(None)
            ann.load(None)

            # Validate count
            self.assertEqual(ann.count(), 2)

            # Test delete
            ann.delete([0])
            self.assertEqual(ann.count(), 1)

            # Close ANN
            ann.close()

    @unittest.skipIf(platform.system() == "Darwin", "SQLite extensions not supported on macOS")
    def testSQLite(self):
        """
        Test SQLite backend
        """

        self.runTests("sqlite")

    @unittest.skipIf(platform.system() == "Darwin", "SQLite extensions not supported on macOS")
    def testSQLiteCustom(self):
        """
        Test SQLite backend with custom settings
        """

        # Test with custom settings
        self.runTests("sqlite", {"sqlite": {"quantize": 1}})
        self.runTests("sqlite", {"sqlite": {"quantize": 8}})

        # Test saving to a new path
        model = self.backend("sqlite")
        expected = model.count() - 1

        # Test save variations
        index = os.path.join(tempfile.gettempdir(), "ann.sqlite")
        new = os.path.join(tempfile.gettempdir(), "ann.sqlite.new")

        # Save new
        model.save(index)

        # Save to same path
        model.save(index)

        # Delete id
        model.delete([0])

        # Save to another path
        model.load(index)
        model.save(new)

        self.assertEqual(model.count(), expected)

    def testTorch(self):
        """
        Test Torch backend
        """

        self.runTests("torch")

    def runTests(self, name, params=None, update=True):
        """
        Runs a series of standard backend tests.

        Args:
            name: backend name
            params: additional config parameters
            update: If append/delete options should be tested
        """

        self.assertEqual(self.backend(name, params).config["backend"], name)
        self.assertEqual(self.save(name, params).count(), 10000)

        if update:
            self.assertEqual(self.append(name, params, 500).count(), 10500)
            self.assertEqual(self.delete(name, params, [0, 1]).count(), 9998)
            self.assertEqual(self.delete(name, params, [100000]).count(), 10000)

        self.assertGreater(self.search(name, params), 0)

    def backend(self, name, params=None, length=10000):
        """
        Test a backend.

        Args:
            name: backend name
            params: additional config parameters
            length: number of rows to generate

        Returns:
            ANN model
        """

        # Generate test data
        data = np.random.rand(length, 240).astype(np.float32)
        self.normalize(data)

        config = {"backend": name, "dimensions": data.shape[1]}
        if params:
            config.update(params)

        model = ANNFactory.create(config)
        model.index(data)

        return model

    def append(self, name, params=None, length=500):
        """
        Appends new data to index.

        Args:
            name: backend name
            params: additional config parameters
            length: number of rows to generate

        Returns:
            ANN model
        """

        # Initial model
        model = self.backend(name, params)

        # Generate test data
        data = np.random.rand(length, 240).astype(np.float32)
        self.normalize(data)

        model.append(data)

        return model

    def delete(self, name, params=None, ids=None):
        """
        Deletes data from index.

        Args:
            name: backend name
            params: additional config parameters
            ids: ids to delete

        Returns:
            ANN model
        """

        # Initial model
        model = self.backend(name, params)
        model.delete(ids)

        return model

    def save(self, name, params=None):
        """
        Test save/load.

        Args:
            name: backend name
            params: additional config parameters

        Returns:
            ANN model
        """

        model = self.backend(name, params)

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "ann")

        # Save and close index
        model.save(index)
        model.close()

        # Reload index
        model.load(index)

        return model

    def search(self, name, params=None):
        """
        Test ANN search.

        Args:
            name: backend name
            params: additional config parameters

        Returns:
            search results
        """

        # Generate ANN index
        model = self.backend(name, params)

        # Generate query vector
        query = np.random.rand(240).astype(np.float32)
        self.normalize(query)

        # Ensure top result has similarity > 0
        return model.search(np.array([query]), 1)[0][0][1]

    def normalize(self, embeddings):
        """
        Normalizes embeddings using L2 normalization. Operation applied directly on array.

        Args:
            embeddings: input embeddings matrix
        """

        # Calculation is different for matrices vs vectors
        if len(embeddings.shape) > 1:
            embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        else:
            embeddings /= np.linalg.norm(embeddings)
