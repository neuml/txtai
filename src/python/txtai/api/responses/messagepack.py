"""
MessagePack module
"""

from io import BytesIO
from typing import Any

import msgpack

from fastapi import Response
from PIL.Image import Image


class MessagePackResponse(Response):
    """
    Encodes responses with MessagePack.
    """

    media_type = "application/msgpack"

    def render(self, content: Any) -> bytes:
        """
        Renders content to the response.

        Args:
            content: input content

        Returns:
            rendered content as bytes
        """

        return msgpack.packb(content, default=MessagePackEncoder())


class MessagePackEncoder:
    """
    Extended MessagePack encoder that converts images to bytes.
    """

    def __call__(self, o):
        # Convert Image to bytes
        if isinstance(o, Image):
            buffered = BytesIO()
            o.save(buffered, format=o.format, quality="keep")
            o = buffered

        # Get bytes from BytesIO
        if isinstance(o, BytesIO):
            o = o.getvalue()

        return o
