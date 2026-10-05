"""
Factory module
"""

from .base import Archive


class ArchiveFactory:
    """
    Methods to create Archive instances.
    """

    @staticmethod
    def create(directory=None):
        """
        Create a new Archive instance.

        Args:
            directory: optional default working directory, otherwise uses a temporary directory

        Returns:
            Archive
        """

        return Archive(directory)
