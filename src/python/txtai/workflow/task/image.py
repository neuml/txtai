"""
ImageTask module
"""

import re

# Conditional import
try:
    from PIL import Image

    PIL = True
except ImportError:
    PIL = False

from .file import FileTask


class ImageTask(FileTask):
    """
    Task that processes image file urls
    """

    def __init__(self, action=None, select=None, unpack=True, column=None, merge="hstack", initialize=None, finalize=None):

        if not PIL:
            raise ImportError('ImageTask is not available - install "workflow" extra to enable')

        super().__init__(action, select, unpack, column, merge, initialize, finalize)

    def accept(self, element):
        # Only accept image files
        return super().accept(element) and re.search(r"\.(gif|bmp|jpg|jpeg|png|webp)$", element.lower())

    def prepare(self, element):
        return Image.open(super().prepare(element))
