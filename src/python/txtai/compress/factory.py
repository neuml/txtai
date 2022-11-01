"""
Factory module
"""

from .tar import Tar
from .zip import Zip


class CompressFactory:
    """
    Methods to create Compress instances.
    """

    @staticmethod
    def create(path):
        """
        Factory method to construct a Compress instance.

        Args:
            path: file path

        Returns:
            Compression
        """

        compression = path.lower().split(".")[-1]
        return Zip() if compression == "zip" else Tar()
