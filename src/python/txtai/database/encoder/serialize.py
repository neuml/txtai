"""
SerializeEncoder module
"""

from ...serialize import SerializeFactory

from .base import Encoder


class SerializeEncoder(Encoder):
    """
    Encodes and decodes objects using the internal serialize package.
    """

    def __init__(self, method):
        # Parent constructor
        super().__init__()

        # Pickle serialization
        self.serializer = SerializeFactory.create(method)

    def encode(self, obj):
        # Pickle object
        return self.serializer.savebytes(obj)

    def decode(self, data):
        # Unpickle to object
        return self.serializer.loadbytes(data)
