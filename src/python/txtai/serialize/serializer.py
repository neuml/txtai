"""
Serializer module
"""

from .errors import SerializeError
from .factory import SerializeFactory


class Serializer:
    """
    Methods to serialize and deserialize data.
    """

    @staticmethod
    def load(path):
        """
        Loads data from path. This method first tries to load the default serialization format.
        If that fails, it will fallback to pickle format for backwards-compatability purposes.

        Note that loading pickle files requires the env variable `ALLOW_PICKLE=True`.

        Args:
            path: data to load

        Returns:
            data
        """

        try:
            return SerializeFactory.create().load(path)
        except SerializeError:
            # Backwards compatible check for pickled data
            return SerializeFactory.create("pickle").load(path)

    @staticmethod
    def save(data, path):
        """
        Saves data to path.

        Args:
            data: data to save
            path: output path
        """

        # Save using default serialization method
        SerializeFactory.create().save(data, path)
