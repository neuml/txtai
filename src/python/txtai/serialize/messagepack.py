"""
MessagePack module
"""

import msgpack
from msgpack import Unpacker
from msgpack.exceptions import ExtraData

from .base import Serialize
from .errors import SerializeError


class MessagePack(Serialize):
    """
    MessagePack serialization.
    """

    def __init__(self, streaming=False, **kwargs):
        # Parent constructor
        super().__init__()

        # Streaming unpacker
        self.streaming = streaming

        # Additional streaming unpacker keyword arguments
        self.kwargs = kwargs

    def loadstream(self, stream):
        try:
            # Support both streaming and non-streaming unpacking of data
            return Unpacker(stream, **self.kwargs) if self.streaming else msgpack.unpack(stream)
        except ExtraData as e:
            raise SerializeError(e) from e

    def savestream(self, data, stream):
        msgpack.pack(data, stream)

    def loadbytes(self, data):
        return msgpack.unpackb(data)

    def savebytes(self, data):
        return msgpack.packb(data)
