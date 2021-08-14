"""
Textractor module
"""

# Conditional import
try:
    from tika import parser

    TIKA = True
except ImportError:
    TIKA = False

from .segmentation import Segmentation


class Textractor(Segmentation):
    """
    Extracts text from files.
    """

    def __init__(self, sentences=False, lines=False, paragraphs=False, minlength=None, join=False):
        if not TIKA:
            raise ImportError('Textractor pipeline is not available - install "pipeline" extra to enable')

        super().__init__(sentences, lines, paragraphs, minlength, join)

    def text(self, text):
        # text is a path to a file
        parsed = parser.from_file(text)
        return parsed["content"]
