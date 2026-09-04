"""
Dense ANN module tests
"""

import os
import platform
import sys
import tempfile
import unittest

from unittest.mock import patch

import numpy as np

from sqlalchemy.dialects.postgresql import BIT
from sqlalchemy.ext.compiler import compiles

from txtai.ann import ANNFactory, ANN
from txtai.serialize import SerializeFactory


# pylint: disable=R0904
class TestDense(unittest.TestCase):
    """
    Dense ANN tests.
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

    def testCustomBackendInvalid(self):
        """
        Test resolving an invalid backend
        """

        with self.assertRaises(ImportError):
            ANNFactory.create({"backend": "pprint.pprint"})

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

    def testFaissBinary(self):
        """
        Test Faiss backend with a binary hash index
        """

        ann = ANNFactory.create({"backend": "faiss", "quantize": 1, "dimensions": 240 * 8, "faiss": {"components": "BHash32"}})

        # Generate and index dummy data
        data = np.random.rand(100, 240).astype(np.uint8)
        ann.index(data)

        # Generate query vector and test search
        query = np.random.rand(240).astype(np.uint8)
        self.assertGreater(ann.search(np.array([query]), 1)[0][0][1], 0)

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
        name = "txtai.ann.dense.faiss"
        module = sys.modules[name]
        del sys.modules[name]
        import txtai.ann.dense.faiss

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

    def testFaissNprobeClamp(self):
        """
        Test Faiss default nprobe clamp
        """

        ann, _ = self.faissmodel(5001, {"sample": 0.01})
        self.assertGreater(round(ann.cells(ann.count()) / 16), ann.backend.nlist)
        self.assertEqual(ann.nprobe(), ann.backend.nlist)

    def testFaissNprobeConfigured(self):
        """
        Test configured Faiss nprobe values
        """

        for nprobe in [1, 3]:
            with self.subTest(nprobe=nprobe):
                ann, _ = self.faissmodel(100, {"components": "IVF2,Flat", "nprobe": nprobe})
                self.assertEqual(ann.nprobe(), nprobe)

    def testFaissNprobeResults(self):
        """
        Test Faiss default nprobe clamp results
        """

        ann, rng = self.faissmodel(5001, {"sample": 0.01})
        queries = rng.random((10, 8), dtype=np.float32)
        self.normalize(queries)

        default = ann.search(queries, 5)
        ann.config["faiss"]["nprobe"] = ann.backend.nlist
        full = ann.search(queries, 5)

        self.assertEqual(default, full)

    def testFaissNprobeSmall(self):
        """
        Test Faiss default nprobe clamp on a small index
        """

        ann, _ = self.faissmodel(100, {"components": "IVF2,Flat"})
        self.assertEqual(ann.backend.nlist, 2)
        self.assertEqual(ann.nprobe(), ann.backend.nlist)

    def testFaissNprobeUnchanged(self):
        """
        Test Faiss default nprobe without a clamp
        """

        ann, _ = self.faissmodel(5001, {"components": "IVF16,Flat"})
        expected = round(ann.cells(ann.count()) / 16)
        self.assertLessEqual(expected, ann.backend.nlist)
        self.assertEqual(ann.nprobe(), expected)

    def testGGML(self):
        """
        Test GGML backend
        """

        self.runTests("ggml")

    def testGGMLDelete(self):
        """
        Test GGML backend deletes
        """

        ann = ANNFactory.create({"backend": "ggml"})

        # Generate and index dummy data
        data = np.random.rand(100, 256).astype(np.float32)
        ann.index(data)

        # Validate count
        self.assertEqual(ann.count(), 100)

        # Test delete
        ann.delete([0])
        self.assertEqual(ann.count(), 99)

        # Deleting the same id again is a no-op for the count
        ann.delete([0])
        self.assertEqual(ann.count(), 99)

        # Save updated index with deletes and reload
        index = os.path.join(tempfile.gettempdir(), "ggml.deletes")
        ann.save(index)
        ann.load(index)
        self.assertEqual(ann.count(), 99)

    def testGGMLEmpty(self):
        """
        Test GGML backend with an empty index
        """

        ann = ANNFactory.create({"backend": "ggml", "dimensions": 240})

        # Index an empty array
        ann.index(np.zeros((0, 240), dtype=np.float32))
        self.assertEqual(ann.count(), 0)

        # Test save and load
        index = os.path.join(tempfile.gettempdir(), "ggml.empty")
        ann.save(index)
        ann.load(index)
        self.assertEqual(ann.count(), 0)

        # Append data to the loaded empty index
        data = np.random.rand(10, 240).astype(np.float32)
        self.normalize(data)
        ann.append(data)
        self.assertEqual(ann.count(), 10)

        # Generate query vector and test search
        query = np.random.rand(240).astype(np.float32)
        self.normalize(query)
        self.assertGreater(ann.search(np.array([query]), 1)[0][0][1], 0)

    def testGGMLQuantization(self):
        """
        Test GGML backend with quantization enabled
        """

        ann = ANNFactory.create({"backend": "ggml", "ggml": {"quantize": "Q4_0"}})

        # Generate and index dummy data
        data = np.random.rand(100, 256).astype(np.float32)
        ann.index(data)

        # Test save and load
        index = os.path.join(tempfile.gettempdir(), "ggml.q4_0.v1")
        ann.save(index)
        ann.load(index)

        # Generate query vector and test search
        query = np.random.rand(256).astype(np.float32)
        self.normalize(query)
        self.assertGreater(ann.search(np.array([query]), 1)[0][0][1], 0)

        # Validate count
        self.assertEqual(ann.count(), 100)

        # Test delete
        ann.delete([0])
        self.assertEqual(ann.count(), 99)

        # Save updated index with deletes and reload
        index = os.path.join(tempfile.gettempdir(), "ggml.q4_0.v2")
        ann.save(index)
        ann.load(index)
        ann.index(data)

    def testGGMLInvalid(self):
        """
        Test invalid GGML configurations
        """

        data = np.random.rand(100, 240).astype(np.float32)

        with self.assertRaises(ValueError):
            ann = ANNFactory.create({"backend": "ggml", "ggml": {"quantize": "NOEXIST", "gpu": False}})
            ann.index(data)

        with self.assertRaises(ValueError):
            ann = ANNFactory.create({"backend": "ggml", "ggml": {"quantize": "Q4_K"}})
            ann.index(data)

    def testGGMLSingleRow(self):
        """
        Test GGML backend with a single row index
        """

        ann = ANNFactory.create({"backend": "ggml", "dimensions": 240})

        # Generate and index a single row
        data = np.random.rand(1, 240).astype(np.float32)
        self.normalize(data)
        ann.index(data)
        self.assertEqual(ann.count(), 1)

        # Test save and load
        index = os.path.join(tempfile.gettempdir(), "ggml.single")
        ann.save(index)
        ann.load(index)
        self.assertEqual(ann.count(), 1)
        self.assertEqual(ann.search(data, 1)[0][0][0], 0)

        # Append a row
        ann.append(data)
        self.assertEqual(ann.count(), 2)

        # Test delete, including an out of range id
        ann.delete([0, 5])
        self.assertEqual(ann.count(), 1)

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

    def testMilvus(self):
        """
        Test milvus-lite backend
        """

        self.runTests("milvus")

    def testMilvusCustom(self):
        """
        Test milvus-lite backend with custom settings
        """

        self.runTests("milvus", {"milvus": {"m": 16}})

        # Test invalid file path handled
        with self.assertRaises(FileNotFoundError):
            ann = ANNFactory.create({"backend": "milvus"})
            ann.load("non-exist-path")

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

    def testNumPySafetensors(self):
        """
        Test NumPy backend with safetensors storage
        """

        ann = ANNFactory.create({"backend": "numpy", "numpy": {"safetensors": True}})

        # Generate and index dummy data
        data = np.random.rand(100, 240).astype(np.float32)
        ann.index(data)

        # Test save and load
        index = os.path.join(tempfile.gettempdir(), "numpy.safetensors")
        ann.save(index)
        ann.load(index)

        # Generate query vector and test search
        query = np.random.rand(240).astype(np.float32)
        self.normalize(query)
        self.assertGreater(ann.search(np.array([query]), 1)[0][0][1], 0)

        # Validate count
        self.assertEqual(ann.count(), 100)

    def testNumPyQuantizeDelete(self):
        """
        Test that deleted rows don't resurface in search results for quantized (hamming) NumPy indexes
        """

        ann = ANNFactory.create({"backend": "numpy", "quantize": 1, "dimensions": 4})

        # Index uint8 vectors
        data = np.array([[1, 0, 0, 0], [0, 255, 0, 0], [0, 0, 255, 0], [0, 0, 0, 255]], dtype=np.uint8)
        ann.index(data)

        # Delete the first row and search with its (former) vector
        ann.delete([0])
        results = ann.search(np.array([data[0]]), 4)[0]

        # Deleted row must score 0 so the caller's score > 0 filter drops it, same as the dot-product path
        self.assertEqual(dict(results).get(0), 0)
        self.assertEqual(ann.count(), 3)

    @patch("sqlalchemy.orm.Query.limit")
    def testPGVector(self, query):
        """
        Test PGVector backend
        """

        # pylint: disable=W0613
        @compiles(BIT, "sqlite")
        def compile_bit_sqlite(type_, compiler, **kw):
            return "BLOB"

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

    def testTorchDelete(self):
        """
        Test Torch backend delete keeps the stored data type for quantized (uint8) and float16 indexes
        """

        for dtype, config in [(np.uint8, {"quantize": 1}), (np.float16, {})]:
            with self.subTest(dtype=dtype.__name__):
                ann = ANNFactory.create({"backend": "torch", "dimensions": 4, **config})

                # Index vectors with a single non-zero value per row
                data = np.array([[1, 0, 0, 0], [0, 255, 0, 0], [0, 0, 255, 0], [0, 0, 0, 255]], dtype=dtype)
                ann.index(data)

                # Delete the first row and search with its (former) vector
                ann.delete([0])
                results = ann.search(np.array([data[0]]), 4)[0]

                # Deleted row must score 0 so the caller's score > 0 filter drops it and the data type must be unchanged
                self.assertEqual(dict(results).get(0), 0)
                self.assertEqual(ann.count(), 3)
                self.assertEqual(str(ann.backend.dtype), f"torch.{dtype.__name__}")

    @unittest.skipIf(platform.system() == "Darwin", "Torch quantization not supported on macOS")
    def testTorchQuantization(self):
        """
        Test Torch backend with quantization enabled
        """

        for qtype in ["fp4", "nf4", "int8"]:
            ann = ANNFactory.create({"backend": "torch", "torch": {"quantize": {"type": qtype}}})

            # Generate and index dummy data
            data = np.random.rand(100, 240).astype(np.float32)
            ann.index(data)

            # Test save and load
            index = os.path.join(tempfile.gettempdir(), f"{qtype}.safetensors")
            ann.save(index)
            ann.load(index)

            # Generate query vector and test search
            query = np.random.rand(240).astype(np.float32)
            self.normalize(query)
            self.assertGreater(ann.search(np.array([query]), 1)[0][0][1], 0)

            # Validate count
            self.assertEqual(ann.count(), 100)

            # Test delete
            ann.delete([0])
            self.assertEqual(ann.count(), 99)

    def testTurboVec(self):
        """
        Test turbovec backend
        """

        self.runTests("turbovec")

    def testZvec(self):
        """
        Test zvec backend
        """

        self.runTests("zvec")

    def testZvecCustom(self):
        """
        Test zvec backend with custom settings
        """

        self.runTests("zvec", {"zvec": {"m": 16}})

        # Test invalid file path handled
        with self.assertRaises(FileNotFoundError):
            ann = ANNFactory.create({"backend": "zvec"})
            ann.load("non-exist-path")

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
        self.assertGreater(self.limit(name, params), 0)

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

    def faissmodel(self, length, settings):
        """
        Builds a seeded Faiss model.

        Args:
            length: number of rows to generate
            settings: Faiss settings

        Returns:
            Faiss model and random number generator
        """

        rng = np.random.default_rng(0)
        data = rng.random((length, 8), dtype=np.float32)
        self.normalize(data)

        model = ANNFactory.create({"backend": "faiss", "dimensions": data.shape[1], "faiss": settings})
        model.index(data)

        return model, rng

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

        # Generate query vector
        query = np.random.rand(240).astype(np.float32)
        self.normalize(query)

        # Save index and ensure it's still searchable
        model.save(index)
        self.assertGreater(model.search(np.array([query]), 1)[0][0][1], 0)

        # Close and reload index
        model.close()
        model.load(index)

        # Ensure reloaded index is searchable
        self.assertGreater(model.search(np.array([query]), 1)[0][0][1], 0)

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

    def limit(self, name, params=None):
        """
        Test ANN limit search.

        Args:
            name: backend name
            params: additional config parameters

        Returns:
            search results
        """

        # Generate ANN index
        model = self.backend(name, params, 50)

        # Generate query vector
        query = np.random.rand(240).astype(np.float32)
        self.normalize(query)

        # Ensure limit being > count doesn't throw an error
        return model.search(np.array([query]), 100)[0][0][1]

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
