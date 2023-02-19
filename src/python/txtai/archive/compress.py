"""
Compress module
"""

import os


class Compress:
    """
    Base class for Compress instances.
    """

    def pack(self, path, output):
        """
        Compresses files in directory path to file output.

        Args:
            path: input directory path
            output: output file
        """

        raise NotImplementedError

    def unpack(self, path, output):
        """
        Extracts all files in path to output.

        Args:
            path: input file path
            output: output directory
        """

        raise NotImplementedError

    def validate(self, directory, path):
        """
        Validates path is under directory.

        Args:
            directory: base directory
            path: path to validate

        Returns:
            True if path is under directory, False otherwise
        """

        directory = os.path.abspath(directory)
        path = os.path.abspath(path)
        prefix = os.path.commonprefix([directory, path])

        return prefix == directory
