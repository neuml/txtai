"""
Compress module tests
"""

import os
import tarfile
import tempfile
import unittest

from zipfile import ZipFile, ZIP_DEFLATED

from txtai.archive import ArchiveFactory, Compress

# pylint: disable=C0411
from utils import Utils


class TestArchive(unittest.TestCase):
    """
    Archive tests.
    """

    def testCompression(self):
        """
        Test compression format is used when provided
        """

        for compression, mode in [("tar.gz", "r:gz"), ("gz", "r:gz"), ("bz2", "r:bz2"), ("xz", "r:xz"), ("tar", "r:")]:
            # Create archive instance
            archive = ArchiveFactory.create()

            # Create file in archive working path
            with open(os.path.join(archive.path(), "test"), "w", encoding="utf-8") as f:
                f.write("test")

            # Save archive to a path without an extension
            path = os.path.join(tempfile.gettempdir(), f"compression-{compression}")
            archive.save(path, compression)

            # Check that the file was written with the requested compression
            with tarfile.open(path, mode) as tar:
                self.assertEqual(tar.getnames(), [".", "./test"])

            # Extract files from archive
            archive = ArchiveFactory.create()
            archive.load(path, compression)

            # Check if file properly extracted
            self.assertTrue(os.path.exists(os.path.join(archive.path(), "test")))

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

    def testInvalidTarLink(self):
        """
        Test invalid tar file with symlinks
        """

        symlink = os.path.join(tempfile.gettempdir(), "link")

        # Remove symlink if it already exists
        try:
            os.remove(symlink)
        except OSError:
            pass

        # Create symlink and add to TAR file
        os.symlink(os.path.join(tempfile.gettempdir(), "noexist"), symlink)

        path = os.path.join(tempfile.gettempdir(), "badtarlink")
        with tarfile.open(path, "w") as tar:
            tar.add(symlink, arcname="l")

        archive = ArchiveFactory.create()

        # Validate error is thrown for file
        with self.assertRaises(IOError):
            archive.load(path, "tar")

    def testInvalidTarPath(self):
        """
        Test invalid tar file with a path outside of base directory
        """

        path = os.path.join(tempfile.gettempdir(), "badtarpath")
        with tarfile.open(path, "w") as tar:
            tar.add(Utils.PATH, arcname="..")

        archive = ArchiveFactory.create()

        # Validate error is thrown for file
        with self.assertRaises(IOError):
            archive.load(path, "tar")

    def testInvalidZipPath(self):
        """
        Test invalid zip file with a path outside of base directory
        """

        path = os.path.join(tempfile.gettempdir(), "badzippath")
        with ZipFile(path, "w", ZIP_DEFLATED) as zfile:
            zfile.write(Utils.PATH + "/article.pdf", arcname="../article.pdf")

        archive = ArchiveFactory.create()

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
