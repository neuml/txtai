"""
FileToHTML module
"""

import os
import re

from subprocess import Popen

# Conditional import
try:
    from tika import detector, parser

    TIKA = True
except ImportError:
    TIKA = False

# Conditional import
try:
    from docling.document_converter import DocumentConverter

    DOCLING = True
except ImportError:
    DOCLING = False

from ..base import Pipeline


class FileToHTML(Pipeline):
    """
    File to HTML pipeline.
    """

    def __init__(self, backend="available"):
        """
        Creates a new File to HTML pipeline.

        Args:
            backend: backend to use to extract content, supports "tika", "docling" or "available" (default) which finds the first available
        """

        # Lowercase backend parameter
        backend = backend.lower() if backend else None

        # Check for available backend
        if backend == "available":
            backend = "tika" if Tika.available() else "docling" if Docling.available() else None

        # Create backend instance
        self.backend = Tika() if backend == "tika" else Docling() if backend == "docling" else None

    def __call__(self, path):
        """
        Converts file at path to HTML. Returns None if no backend is available.

        Args:
            path: input file path

        Returns:
            html if a backend is available, otherwise returns None
        """

        return self.backend(path) if self.backend else None


class Tika:
    """
    File to HTML conversion via Apache Tika.
    """

    @staticmethod
    def available():
        """
        Checks if a Java executable is available and Tika is installed.

        Returns:
            True if Java is available and Tika is installed, False otherwise
        """

        # Get path to Java executable
        path = os.environ.get("TIKA_JAVA", "java")

        # pylint: disable=R1732,W0702,W1514
        # Check if Java binary is available on path
        try:
            _ = Popen(path, stdout=open(os.devnull, "w"), stderr=open(os.devnull, "w"))
        except:
            return False

        # Return True if Java is available AND Tika is installed
        return TIKA

    def __init__(self):
        """
        Creates a new Tika instance.
        """

        if not Tika.available():
            raise ImportError('Tika engine is not available - install "pipeline" extra to enable. Also check that Java is available.')

    def __call__(self, path):
        """
        Parses content to HTML.

        Args:
            path: file path

        Returns:
            html
        """

        # Skip parsing if input is plain text or HTML
        mimetype = detector.from_file(path)
        if mimetype in ("text/plain", "text/html", "text/xhtml"):
            return None

        # Parse content to HTML
        parsed = parser.from_file(path, xmlContent=True)
        return parsed["content"]


class Docling:
    """
    File to HTML conversion via Docling.
    """

    @staticmethod
    def available():
        """
        Checks if Docling is available.

        Returns:
            True if Docling is available, False otherwise
        """

        return DOCLING

    def __init__(self):
        """
        Creates a new Docling instance.
        """

        if not Docling.available():
            raise ImportError('Docling engine is not available - install "pipeline" extra to enable')

        self.converter = DocumentConverter()

    def __call__(self, path):
        """
        Parses content to HTML.

        Args:
            path: file path

        Returns:
            html
        """

        # Skip parsing if input is HTML
        if self.ishtml(path):
            return None

        # Parse content to HTML
        html = self.converter.convert(path).document.export_to_html(html_head="<head/>")

        # Normalize HTML and return
        return self.normalize(html)

    def ishtml(self, path):
        """
        Detects if this file looks like HTML.

        Args:
            path: file path

        Returns:
            True if this is HTML
        """

        with open(path, "rb") as f:
            # Read first 1024 bytes, ignore encoding errors and strip leading/trailing whitespace
            content = f.read(1024)
            content = content.decode("ascii", errors="ignore").lower().strip()

            # Check for HTML
            return re.search(r"<!doctype\s+html|<html|<head|<body", content)

    def normalize(self, html):
        """
        Applies normalization rules to make HTML consistent with other text extraction backends.

        Args:
            html: input html

        Returns:
            normalized html
        """

        # Wrap content with a body tag
        html = html.replace("<head/>", "<head/><body>").replace("</html>", "</body></html>")

        # Remove bullets from list items
        html = re.sub(r"<li>\xb7 ", r"<li>", html)

        # Add spacing between paragraphs
        return html.replace("</p>", "</p><p/>")
