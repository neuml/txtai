"""
Archive module
"""

import os

from tempfile import TemporaryDirectory

from .tar import Tar
from .zip import Zip


class Archive:
    """
    Base class for archive instances.
    """

    def __init__(self, directory=None):
        """
        Creates a new archive instance.

        Args:
            directory: directory to use as working directory, defaults to a temporary directory
        """

        self.directory = directory

    def isarchive(self, path):
        """
        Checks if path is an archive file based on the extension.

        Args:
            path: path to check

        Returns:
            True if the path ends with an archive extension, False otherwise
        """

        return path and any(path.lower().endswith(extension) for extension in [".tar.bz2", ".tar.gz", ".tar.xz", ".zip"])

    def path(self):
        """
        Gets the current working directory for this archive instance.

        Returns:
            archive working directory
        """

        # Default to a temporary directory. All files created in this directory will be deleted
        # when this archive instance goes out of scope.
        if not self.directory:
            # pylint: disable=R1732
            self.directory = TemporaryDirectory()

        return self.directory.name if isinstance(self.directory, TemporaryDirectory) else self.directory

    def load(self, path, compression=None):
        """
        Extracts file at path to archive working directory.

        Args:
            path: path to archive file
            compression: compression format, infers from path if not provided
        """

        # Unpack compressed file
        compress = self.create(path, compression)
        compress.unpack(path, self.path())

    def save(self, path, compression=None):
        """
        Archives files in archive working directory to file at path.

        Args:
            path: path to archive file
            compression: compression format, infers from path if not provided
        """

        # Create output directory, if necessary
        output = os.path.dirname(path)
        if output:
            os.makedirs(output, exist_ok=True)

        # Pack into compressed file
        compress = self.create(path, compression)
        compress.pack(self.path(), path)

    def create(self, path, compression):
        """
        Method to construct a Compress instance.

        Args:
            path: file path
            compression: compression format, infers using file extension if not provided

        Returns:
            Compress
        """

        # Infer compression format from path if not provided
        compression = compression if compression else path.lower().split(".")[-1]

        # Create compression instance
        return Zip() if compression == "zip" else Tar()
