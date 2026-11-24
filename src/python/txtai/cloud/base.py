"""
Cloud module
"""

import os

from ..archive import ArchiveFactory


class Cloud:
    """
    Base class for cloud providers. Cloud providers sync content between local and remote storage.
    """

    def __init__(self, config):
        """
        Creates a new cloud connection.

        Args:
            config: cloud configuration
        """

        self.config = config

    def exists(self, path=None):
        """
        Checks if path exists in cloud. If path is None, this method checks if the container exists.

        Args:
            path: path to check

        Returns:
            True if path or container exists, False otherwise
        """

        return self.metadata(path) is not None

    def metadata(self, path=None):
        """
        Returns metadata for path from cloud. If path is None, this method returns metadata
        for container.

        Args:
            path: retrieve metadata for this path

        Returns:
            path or container metadata if available, otherwise returns None
        """

        raise NotImplementedError

    def load(self, path=None):
        """
        Retrieves content from cloud and stores locally. If path is empty, this method retrieves
        all content in the container.

        Args:
            path: path to retrieve

        Returns:
            local path which can be different than input path
        """

        raise NotImplementedError

    def save(self, path):
        """
        Sends local content stored in path to cloud.

        Args:
            path: local path to sync
        """

        raise NotImplementedError

    def isarchive(self, path):
        """
        Check if path is an archive file.

        Args:
            path: path to check

        Returns:
            True if path ends with an archive extension, false otherwise
        """

        return ArchiveFactory.create().isarchive(path)

    def listfiles(self, path):
        """
        Lists files in path. If path is a directory, this method returns all files in path including
        files in subdirectories. If path is a file, this method returns a single element list containing path.

        Args:
            path: path to list

        Returns:
            List of files
        """

        # List all files if path is a directory, including files in subdirectories (e.g. subindexes)
        if os.path.isdir(path):
            return [os.path.join(root, f) for root, _, files in sorted(os.walk(path)) for f in sorted(files)]

        # Path is a file
        return [path]

    def filename(self, path, f):
        """
        Gets the remote name for a file returned by listfiles. Files in a directory are named using
        their path relative to that directory. Otherwise, the file name is used.

        Args:
            path: path passed to listfiles
            f: file path

        Returns:
            remote file name
        """

        name = os.path.relpath(f, path) if os.path.isdir(path) else os.path.basename(f)
        return name.replace(os.sep, "/")
