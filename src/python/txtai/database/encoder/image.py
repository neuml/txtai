"""
ImageEncoder module
"""

from io import BytesIO

from .base import Encoder

# Conditional import
try:
    from PIL import Image

    PIL = True
except ImportError:
    PIL = False


class ImageEncoder(Encoder):
    """
    Encodes and decodes Image objects as compressed binary content, using the original image's algorithm.
    """

    def __init__(self):
        """
        Creates a new ImageEncoder.
        """

        if not PIL:
            raise ImportError('ImageEncoder is not available - install "database" extra to enable')

    def encode(self, obj):
        # Create byte stream
        output = BytesIO()

        # Write image to byte stream
        obj.save(output, format=obj.format, quality="keep")

        # Return byte array
        return output.getvalue()

    def decode(self, data):
        # Return a PIL image
        return Image.open(BytesIO(data)) if data else None
