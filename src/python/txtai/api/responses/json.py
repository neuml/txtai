"""
JSON module
"""

import base64
import json

from io import BytesIO
from typing import Any

import fastapi.responses

from PIL.Image import Image


class JSONEncoder(json.JSONEncoder):
    """
    Extended JSONEncoder that serializes images and byte streams as base64 encoded text.
    """

    def default(self, o):
        # Convert Image to BytesIO
        if isinstance(o, Image):
            buffered = BytesIO()
            o.save(buffered, format=o.format, quality="keep")
            o = buffered

        # Unpack bytes from BytesIO
        if isinstance(o, BytesIO):
            o = o.getvalue()

        # Base64 encode bytes instances
        if isinstance(o, bytes):
            return base64.b64encode(o).decode("utf-8")

        # Default handler
        return super().default(o)


class JSONResponse(fastapi.responses.JSONResponse):
    """
    Extended JSONResponse that serializes images and byte streams as base64 encoded text.
    """

    def render(self, content: Any) -> bytes:
        """
        Renders content to the response.

        Args:
            content: input content

        Returns:
            rendered content as bytes
        """

        return json.dumps(content, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":"), cls=JSONEncoder).encode("utf-8")
