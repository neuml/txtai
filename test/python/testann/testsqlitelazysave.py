"""
SQLite lazy-load persistence tests.
"""

import gc
import os
import platform
import tempfile
import unittest

import numpy as np

from txtai import Embeddings
from txtai.ann import ANNFactory


@unittest.skipIf(platform.system() == "Darwin", "SQLite extensions not supported on macOS")
class TestSQLiteLazySave(unittest.TestCase):
    """
    Saving a loaded index must not require an earlier read or write.
    """

    def testSaveImmediately(self):
        """
        Save a freshly loaded ANN to the same or a new path in each storage mode.
        """

        for quantize in (None, 1, 8):
            for copy in (False, True):
                with self.subTest(quantize=quantize, copy=copy):
                    source, target = self.paths(copy)
                    config = {"backend": "sqlite", "dimensions": 8, "sqlite": {"quantize": quantize, "table": "custom_vectors"}}
                    original = self.ann(config)
                    data = self.vectors()
                    original.index(data)
                    expected = original.search(data, 3)
                    original.save(source)
                    original.close()

                    loaded = self.ann(original.config)
                    loaded.load(source)
                    self.assertIsNone(loaded.connection)
                    loaded.save(target)
                    loaded.close()

                    saved = self.ann(loaded.config)
                    saved.load(target)
                    self.assertEqual(saved.count(), len(data))
                    self.assertEqual(saved.search(data, 3), expected)
                    saved.close()

                    # Copying must leave the source readable and unchanged.
                    original.load(source)
                    self.assertEqual(original.search(data, 3), expected)
                    original.close()

    def testEmbeddingsSaveImmediately(self):
        """
        The public load/save API also works without an intervening query.
        """

        for quantize in (None, 1, 8):
            for copy in (False, True):
                with self.subTest(quantize=quantize, copy=copy):
                    source, target = self.paths(copy)
                    data = self.vectors()
                    original = Embeddings({"method": "external", "backend": "sqlite", "sqlite": {"quantize": quantize}})
                    self.addCleanup(original.close)
                    original.index([(str(i), row, None) for i, row in enumerate(data)])
                    expected = original.batchsearch(data, 3)
                    original.save(source)
                    original.close()

                    loaded = Embeddings()
                    self.addCleanup(loaded.close)
                    loaded.load(source)
                    self.assertIsNone(loaded.ann.connection)
                    loaded.save(target)
                    loaded.close()

                    saved = Embeddings()
                    self.addCleanup(saved.close)
                    saved.load(target)
                    self.assertEqual(saved.count(), len(data))
                    self.assertEqual(saved.batchsearch(data, 3), expected)
                    saved.close()

    def testSaveAfterChanges(self):
        """
        Existing connections still save pending deletes and appends.
        """

        for quantize in (None, 1, 8):
            for copy in (False, True):
                with self.subTest(quantize=quantize, copy=copy):
                    source, target = self.paths(copy)
                    config = {"backend": "sqlite", "dimensions": 8, "sqlite": {"quantize": quantize}}
                    data = self.vectors()
                    original = self.ann(config)
                    original.index(data)
                    original.save(source)
                    original.close()

                    loaded = self.ann(original.config)
                    loaded.load(source)
                    loaded.delete([0])
                    loaded.append(data[1:2])
                    expected = loaded.search(data, 3)
                    loaded.save(target)
                    loaded.close()

                    saved = self.ann(loaded.config)
                    saved.load(target)
                    self.assertEqual(saved.count(), 3)
                    self.assertEqual(saved.search(data, 3), expected)
                    self.assertEqual({i for i, _ in saved.search(data[:1], 3)[0]}, {1, 2, 3})
                    saved.close()

    def paths(self, copy):
        """
        Create temporary paths and arrange cleanup after closing connections.

        Args:
            copy: whether the target differs from the source

        Returns:
            source and target paths
        """

        # Cleanup must run after the separately registered connection cleanups.
        directory = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(directory.cleanup)
        # Collect connections left in failed-copy tracebacks before removing files.
        self.addCleanup(gc.collect)
        source = os.path.join(directory.name, "source")
        return source, os.path.join(directory.name, "target") if copy else source

    def ann(self, config):
        """
        Create an ANN with registered cleanup.

        Args:
            config: index configuration

        Returns:
            ANN instance
        """

        model = ANNFactory.create(config.copy())
        self.addCleanup(model.close)
        return model

    def vectors(self):
        """
        Build normalized vectors with distinct binary encodings.

        Returns:
            float32 vector matrix
        """

        data = np.ones((3, 8), dtype=np.float32)
        data[1, :2] = -1
        data[2, :4] = -1
        return data / np.sqrt(np.float32(8))
