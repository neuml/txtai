"""
Archive module
"""

import os
import tarfile

from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

from .cloud import Cloud


class Archive:
    """
    Methods to load and save archive files.
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

        return any(path.lower().endswith(extension) for extension in [".tar.bz2", ".tar.gz", ".tar.xz", ".zip"])

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

        return self.directory.name

    def exists(self, path, config):
        """
        Checks if file at path exists.

        Args:
            path: path to archive file
            config: additional configuration

        Returns:
            True if path exists, False otherwise
        """

        # Check if archive exists in cloud storage
        cloud = self.cloud(config)
        if cloud:
            return cloud.exists(path)

        # Check if archive exists locally
        return os.path.exists(path)

    def load(self, path, config):
        """
        Extracts file at path to archive working directory.

        Args:
            path: path to archive file
            config: additional configuration
        """

        # Retrieve archive from cloud storage, if necessary
        cloud = self.cloud(config)
        if cloud:
            cloud.load(path)

        # Get compression type
        compression = self.compression(path)

        # Zip files
        if compression == "zip":
            with ZipFile(path, "r") as zfile:
                zfile.extractall(self.path())

        # Tar files
        else:
            with tarfile.open(path, f"r:{compression}") as tar:
                tar.extractall(self.path())

    def save(self, path, config):
        """
        Archives files in archive working directory to file at path.

        Args:
            path: path to archive file
            config: additional configuration
        """

        # Get compression type
        compression = self.compression(path)

        # Create output directory, if necessary
        output = os.path.dirname(path)
        if output:
            os.makedirs(output, exist_ok=True)

        # Zip files
        if compression == "zip":
            with ZipFile(path, "w", ZIP_DEFLATED) as zfile:
                for root, _, files in sorted(os.walk(self.path())):
                    for f in files:
                        zfile.write(os.path.join(root, f), arcname=f)

        # Tar files
        else:
            with tarfile.open(path, f"w:{compression}") as tar:
                tar.add(self.path(), arcname=".")

        # Save file to cloud storage, if necessary
        cloud = self.cloud(config)
        if cloud:
            cloud.save(path)

    def compression(self, path):
        """
        Gets compression type for path.

        Args:
            path: path to archive file

        Returns:
            compression type
        """

        return path.lower().split(".")[-1]

    def cloud(self, config):
        """
        Gets a cloud storage instance, if specified in config.

        Args:
            config: cloud storage configuration

        Returns:
            Cloud storage instance
        """

        # Create cloud storage connection, if necessary
        return Cloud(config) if config and "provider" in config else None
