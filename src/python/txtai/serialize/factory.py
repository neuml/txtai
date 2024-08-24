"""
Factory module
"""

from .messagepack import MessagePack
from .pickle import Pickle


class SerializeFactory:
    """
    Methods to create data serializers.
    """

    @staticmethod
    def create(method=None, **kwargs):
        """
        Creates a new Serialize instance.

        Args:
            method: serialization method
            kwargs: additional keyword arguments to pass to serialize instance
        """

        # Pickle serialization
        if method == "pickle":
            return Pickle(**kwargs)

        # Default serialization
        return MessagePack(**kwargs)
