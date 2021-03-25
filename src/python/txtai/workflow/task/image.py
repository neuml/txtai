"""
ImageTask module
"""

import os
import re

from urllib.parse import urlparse

from PIL import Image

from .file import FileTask


class ImageTask(FileTask):
    """
    Task that processes image file urls
    """

    def accept(self, element):
        # Only accept file URLs
        return super().accept(element) and re.search(r"\.(gif|bmp|jpg|jpeg|png|webp)$", element.lower())

    def prepare(self, element):
        # Read image data
        url = urlparse(element)
        path = os.path.abspath(os.path.join(url.netloc, url.path))

        return Image.open(path)
