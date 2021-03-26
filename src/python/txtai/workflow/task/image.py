"""
ImageTask module
"""

import re

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
        return Image.open(super().prepare(element))
