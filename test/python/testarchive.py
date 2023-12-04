"""
Compress module tests
"""

import os
import tarfile
import tempfile
import unittest

from zipfile import ZipFile, ZIP_DEFLATED

from txtai.archive import ArchiveFactory, Compress

# pylint: disable = C0411
from utils import Utils


class TestArchive(unittest.TestCase):
    """
    Archive tests.
    """

    def testDirectory(self):
        """
        Test directory included in compressed files
        """

        for extension in ["tar", "zip"]:
            # Create archive instance
            archive = ArchiveFactory.create()

            # Create subdirectory in archive working path
            path = os.path.join(archive.path(), "dir")
            os.makedirs(path, exist_ok=True)

            # Create file in archive working path
            with open(os.path.join(path, "test"), "w", encoding="utf-8") as f:
                f.write("test")

            # Save archive
            path = os.path.join(tempfile.gettempdir(), f"subdir.{extension}")
            archive.save(path)

            # Extract files from archive
            archive = ArchiveFactory.create()
            archive.load(path)

            # Check if file properly extracted
            path = os.path.join(archive.path(), "dir", "test")
            self.assertTrue(os.path.exists(path))

    def testInvalidTar(self):
        """
        Test invalid tar file
        """

        path = os.path.join(tempfile.gettempdir(), "badtar")
        with tarfile.open(path, "w") as tar:
            tar.add(Utils.PATH, arcname="..")

        archive = ArchiveFactory.create(path)

        # Validate error is thrown for file
        with self.assertRaises(IOError):
            archive.load(path, "tar")

    def testInvalidZip(self):
        """
        Test invalid zip file
        """

        path = os.path.join(tempfile.gettempdir(), "badzip")
        with ZipFile(path, "w", ZIP_DEFLATED) as zfile:
            zfile.write(Utils.PATH + "/article.pdf", arcname="../article.pdf")

        archive = ArchiveFactory.create(path)

        # Validate error is thrown for file
        with self.assertRaises(IOError):
            archive.load(path, "zip")

    def testNotImplemented(self):
        """
        Test exceptions for non-implemented methods
        """

        compress = Compress()

        self.assertRaises(NotImplementedError, compress.pack, None, None)
        self.assertRaises(NotImplementedError, compress.unpack, None, None)
