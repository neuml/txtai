"""
Textractor module
"""

from tika import parser

from .segmentation import Segmentation


class Textractor(Segmentation):
    """
    Extracts text from files.
    """

    def text(self, text):
        # text is a path to a file
        parsed = parser.from_file(text)
        return parsed["content"]
