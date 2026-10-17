"""
SQLite-backed index save path tests.
"""

import os
import platform
import tempfile
import unittest

import numpy as np

from txtai import Embeddings
from txtai.ann import ANNFactory


class TestSavePaths(unittest.TestCase):
    """
    Saving through a filesystem alias must keep the active connection usable.
    """

    def testEmbeddingsPaths(self):
        """
        Exercise content and standalone keyword databases without model downloads.
        """

        for content in (True, False):
            for mode in ("same", "dot", "relative", "symlink", "hardlink", "new", "existing"):
                with self.subTest(content=content, mode=mode), tempfile.TemporaryDirectory(dir=os.getcwd()) as directory:
                    path = os.path.join(directory, "index")
                    embeddings = Embeddings(keyword=True, content=content)
                    try:
                        embeddings.index([("a", ["alpha", "document"], None)])
                        embeddings.save(path)
                        target = self.destination(path, mode)
                        embeddings.save(target)
                        embeddings.upsert([("b", ["beta", "document"], None)])
                        embeddings.save(path)
                    finally:
                        embeddings.close()

                    # Both the original and the requested destination remain loadable.
                    for saved in (path, target):
                        restored = Embeddings()
                        try:
                            restored.load(saved)
                            expected = 1 if mode in ("new", "existing") and saved == target else 2
                            self.assertEqual(restored.count(), expected)
                            self.assertTrue(restored.search(["alpha"]))
                        finally:
                            restored.close()

    @unittest.skipIf(platform.system() == "Darwin", "SQLite extensions not supported on macOS")
    def testSQLiteANNPaths(self):
        """
        Exercise real sqlite-vec databases, including pending changes and lazy loads.
        """

        for quantize in (None, 1, 8):
            for mode in ("same", "dot", "relative", "symlink", "hardlink", "new", "existing"):
                with self.subTest(quantize=quantize, mode=mode), tempfile.TemporaryDirectory(dir=os.getcwd()) as directory:
                    path = os.path.join(directory, "index")
                    ann = ANNFactory.create({"backend": "sqlite", "dimensions": 8, "sqlite": {"quantize": quantize}})
                    data = np.ones((2, 8), dtype=np.float32)
                    data[1, :4] = -1
                    try:
                        ann.index(data[:1])
                        ann.save(path)
                        target = self.destination(path, mode)
                        ann.close()
                        ann.load(path)
                        ann.save(target)
                        ann.append(data[1:])
                        ann.save(target)
                        ann.append(data[:1])
                        ann.save(path)
                    finally:
                        ann.close()

                    for saved in (path, target):
                        restored = ANNFactory.create(ann.config)
                        try:
                            restored.load(saved)
                            expected = 2 if mode in ("new", "existing") and saved == target else 3
                            self.assertEqual(restored.count(), expected)
                            self.assertEqual(len(restored.search(data[:1], 10)[0]), restored.count())
                        finally:
                            restored.close()

    def destination(self, path, mode):
        """
        Create equivalent filesystem paths or genuine Save As destinations.
        """

        parent, name = os.path.split(path)
        paths = {"same": path, "dot": os.path.join(parent, ".", name), "relative": os.path.relpath(path)}
        if mode in paths:
            return paths[mode]

        target = os.path.join(parent, "other")
        if mode == "symlink":
            try:
                os.symlink(path, target, target_is_directory=os.path.isdir(path))
            except (OSError, NotImplementedError) as error:
                self.skipTest(str(error))
        elif mode == "hardlink":
            try:
                if os.path.isdir(path):
                    os.mkdir(target)
                    for filename in os.listdir(path):
                        os.link(os.path.join(path, filename), os.path.join(target, filename))
                else:
                    os.link(path, target)
            except (OSError, NotImplementedError) as error:
                self.skipTest(str(error))
        elif mode == "existing":
            if os.path.isdir(path):
                os.mkdir(target)
                for filename in os.listdir(path):
                    with open(os.path.join(target, filename), "wb") as output:
                        output.write(b"replace me")
            else:
                with open(target, "wb") as output:
                    output.write(b"replace me")
        return target
