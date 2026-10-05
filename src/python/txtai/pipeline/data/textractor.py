"""
Textractor module
"""

import os
import tempfile

from .filetohtml import FileToHTML
from .htmltomd import HTMLToMarkdown
from .safeopen import SafeOpen
from .segmentation import Segmentation


class Textractor(Segmentation):
    """
    Extracts text from files.
    """

    # pylint: disable=R0913
    def __init__(
        self,
        sentences=False,
        lines=False,
        paragraphs=False,
        minlength=None,
        join=False,
        sections=False,
        cleantext=True,
        chunker=None,
        headers=None,
        backend="available",
        safeopen=False,
        **kwargs,
    ):
        super().__init__(sentences, lines, paragraphs, minlength, join, sections, cleantext, chunker, **kwargs)

        # Get backend parameter - handle legacy tika flag
        backend = "tika" if "tika" in kwargs and kwargs["tika"] else None if "tika" in kwargs else backend

        # File to HTML pipeline
        self.html = FileToHTML(backend) if backend else None

        # HTML to Markdown pipeline
        self.markdown = HTMLToMarkdown(self.paragraphs, self.sections)

        # Safe open mode. When set only local temp urls (or a specified directory) and non-private URLs are supported
        self.safeopen = SafeOpen(headers, safeopen)

    def text(self, text):
        # Check if text is a valid file path or url
        path, exists = self.safeopen.valid(text)

        if not path:
            # Not a valid file path, treat input as data
            html = text

        elif self.html:
            # Use FileToHTML pipeline, if available
            # Retrieve remote file, if necessary
            path = path if exists else self.download(path)

            # Parse content to HTML
            html = self.html(path)

            # FiletoHTML pipeline returns None when input is already HTML
            html = html if html else self.safeopen.retrieve(path)

            # Delete temporary file
            if not exists:
                os.remove(path)

        else:
            # Read data from url/path
            html = self.safeopen.retrieve(path)

        # HTML to Markdown
        return self.markdown(html)

    def download(self, url):
        """
        Downloads content of url to a temporary file.

        Args:
            url: input url

        Returns:
            temporary file path
        """

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as output:
            path = output.name

            # Retrieve and write data to temporary file
            output.write(self.safeopen.retrieve(url))

        return path
