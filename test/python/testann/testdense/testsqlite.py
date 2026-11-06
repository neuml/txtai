"""
SQLite ANN module tests
"""

import os
import platform
import tempfile
import unittest

import numpy as np

from txtai.ann import ANNFactory

from .base import DenseTest


class TestSQLite(DenseTest):
    """
    SQLite ANN tests.
    """

    @unittest.skipIf(platform.system() == "Darwin", "SQLite extensions not supported on macOS")
    def testSQLite(self):
        """
        Test SQLite backend
        """

        self.runTests("sqlite")
        self.deletenumpy("sqlite")

    @unittest.skipIf(platform.system() == "Darwin", "SQLite extensions not supported on macOS")
    def testSQLiteBinaryScores(self):
        """
        Binary similarity is the fraction of matching bits, not one minus their distance.
        """

        ann = ANNFactory.create({"backend": "sqlite", "dimensions": 8, "sqlite": {"quantize": 1}})
        self.addCleanup(ann.close)
        data = np.ones((4, 8), dtype=np.float32)
        data[1, :1], data[2, :4], data[3, :] = -1, -1, -1
        ann.index(data)
        self.assertEqual(ann.search(data[:1], 4)[0], [(0, 1.0), (1, 0.875), (2, 0.5), (3, 0.0)])

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

    @unittest.skipIf(platform.system() == "Darwin", "SQLite extensions not supported on macOS")
    def testSQLiteQuantizeDisabled(self):
        """
        Test that quantize: false disables SQLite storage quantization
        """

        for quantize, expected in [(False, None), (True, 8), (1, 1)]:
            with self.subTest(quantize=quantize):
                ann = ANNFactory.create({"backend": "sqlite", "dimensions": 4, "sqlite": {"quantize": quantize}})
                self.assertEqual(ann.quantize, expected)

    @unittest.skipIf(platform.system() == "Darwin", "SQLite extensions not supported on macOS")
    @unittest.skipIf(os.name == "nt", "SQLite copy skipped on Windows due to file locking")
    def testSQLiteSaveExistingPath(self):
        """
        Test saving a SQLite index to an existing path and overwriting the existing database
        """

        # Test saving to a new path
        model = self.backend("sqlite")

        # Test save variations
        index = os.path.join(tempfile.gettempdir(), "ann.sqlite.existing")

        # Save new
        model.save(index)

        # Test saving to a new path
        model = self.backend("sqlite")
        expected = model.count() - 1

        # Delete id
        model.delete([0])

        # Save to same path
        model.save(index)

        self.assertEqual(model.count(), expected)

    @unittest.skipIf(platform.system() == "Darwin", "SQLite extensions not supported on macOS")
    def testSQLiteSaveNewPath(self):
        """
        Test saving a loaded and modified SQLite index to a new path, then loading the new copy
        """

        for quantize in [None, 1, 8]:
            params = {"sqlite": {"quantize": quantize}} if quantize else None

            index = os.path.join(tempfile.gettempdir(), f"ann.sqlite.load.{quantize}")
            new = os.path.join(tempfile.gettempdir(), f"ann.sqlite.load.{quantize}.new")

            # Build and save index
            model = self.backend("sqlite", params, 500)
            model.save(index)
            model.close()

            # Load index, modify and save to a new path
            model = ANNFactory.create(model.config)
            model.load(index)
            model.delete([0])
            model.append(np.random.rand(10, 240).astype(np.float32))
            model.save(new)
            model.close()

            # Load new copy and check that it has all the changes
            model = ANNFactory.create(model.config)
            model.load(index)
            self.assertEqual(model.count(), 500)
            model.load(new)
            self.assertEqual(model.count(), 509)
            self.assertEqual(len(model.search(np.random.rand(1, 240).astype(np.float32), 10)[0]), 10)
            model.close()

    @unittest.skipIf(platform.system() == "Darwin", "SQLite extensions not supported on macOS")
    def testSQLiteSaveWithoutQuery(self):
        """
        Saving after load must not require an earlier query to open the connection.
        """

        for quantize in (None, 1, 8):
            with self.subTest(quantize=quantize), tempfile.TemporaryDirectory() as directory:
                source, target = (os.path.join(directory, name) for name in ("source", "target"))
                ann = ANNFactory.create({"backend": "sqlite", "dimensions": 8, "sqlite": {"quantize": quantize}})
                data = np.ones((2, 8), dtype=np.float32)
                data[1, :4] = -1
                try:
                    ann.index(data)
                    expected = ann.search(data[:1], 2)
                    ann.save(source)
                    for path in (source, target):
                        ann.close()
                        ann.load(source)
                        self.assertIsNone(ann.connection)
                        ann.save(path)
                        ann.close()
                        ann.load(path)
                        self.assertEqual(ann.count(), 2)
                        self.assertEqual(ann.search(data[:1], 2), expected)
                finally:
                    ann.close()
