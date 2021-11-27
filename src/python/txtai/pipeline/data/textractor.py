"""
Textractor module
"""

import contextlib
import os

from subprocess import Popen
from urllib.request import urlopen

# Conditional import
try:
    from bs4 import BeautifulSoup
    from tika import parser

    TIKA = True
except ImportError:
    TIKA = False

from .segmentation import Segmentation


class Textractor(Segmentation):
    """
    Extracts text from files.
    """

    def __init__(self, sentences=False, lines=False, paragraphs=False, minlength=None, join=False, tika=True):
        if not TIKA:
            raise ImportError('Textractor pipeline is not available - install "pipeline" extra to enable')

        super().__init__(sentences, lines, paragraphs, minlength, join)

        # Determine if Tika (default if Java is available) or Beautiful Soup should be used
        # Beautiful Soup only supports HTML, Tika supports a wide variety of file formats, including HTML.
        self.tika = self.checkjava() if tika else False

    def text(self, text):
        # Use Tika if available
        if self.tika:
            # Format file urls as local file paths
            text = text.replace("file://", "")

            # text is a path to a file
            parsed = parser.from_file(text)
            return parsed["content"]

        # Fallback to Beautiful Soup
        text = f"file://{text}" if os.path.exists(text) else text
        with contextlib.closing(urlopen(text)) as connection:
            text = connection.read()

        soup = BeautifulSoup(text, features="html.parser")
        return soup.get_text()

    def checkjava(self, path=None):
        """
        Checks if a Java executable is available for Tika.

        Args:
            path: path to java executable

        Returns:
            True if Java is available, False otherwise
        """

        # Get path to java executable if path not set
        if not path:
            path = os.getenv("TIKA_JAVA", "java")

        # pylint: disable=R1732,W0702,W1514
        # Check if java binary is available on path
        try:
            _ = Popen(path, stdout=open(os.devnull, "w"), stderr=open(os.devnull, "w"))
        except:
            return False

        return True
