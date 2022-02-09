"""
Encoder module
"""

from io import BytesIO


class Encoder:
    """
    Encodes and decodes object content. The base encoder works only with byte arrays. It can be extended to encode different datatypes.
    """

    def encode(self, obj):
        """
        Encodes an object to a byte array using the encoder.

        Returns:
            encoded object as a byte array
        """

        return obj

    def decode(self, data):
        """
        Decodes input byte array into an object using this encoder.

        Args:
            data: encoded data

        Returns:
            decoded object
        """

        return BytesIO(data) if data else None
