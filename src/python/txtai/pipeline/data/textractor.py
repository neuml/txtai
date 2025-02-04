"""
Textractor module
"""

import contextlib
import os
import tempfile

from urllib.parse import urlparse
from urllib.request import urlopen, Request

from .filetohtml import FileToHTML
from .htmltomd import HTMLToMarkdown
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
        **kwargs
    ):
        super().__init__(sentences, lines, paragraphs, minlength, join, sections, cleantext, chunker, **kwargs)

        # Get backend parameter - handle legacy tika flag
        backend = "tika" if "tika" in kwargs and kwargs["tika"] else None if "tika" in kwargs else backend

        # File to HTML pipeline
        self.html = FileToHTML(backend) if backend else None

        # HTML to Markdown pipeline
        self.markdown = HTMLToMarkdown(self.paragraphs, self.sections)

        # HTTP headers
        self.headers = headers if headers else {}

    def text(self, text):
        # Check if text is a valid file path or url
        path, exists = self.valid(text)

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
            html = html if html else self.retrieve(path)

            # Delete temporary file
            if not exists:
                os.remove(path)

        else:
            # Read data from url/path
            html = self.retrieve(path)

        # HTML to Markdown
        return self.markdown(html)

    def valid(self, path):
        """
        Checks if path is a valid local file or web url. Returns path if valid along with a flag
        denoting if the path exists locally.

        Args:
            path: path to check

        Returns:
            (path, exists)
        """

        # Convert file urls to local paths
        path = path.replace("file://", "")

        # Check if this is a local file path or local file url
        exists = os.path.exists(path)

        # Consider local files and HTTP urls valid
        return (path if exists or urlparse(path).scheme in ("http", "https") else None, exists)

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
            output.write(self.retrieve(url))

        return path

    def retrieve(self, url):
        """
        Retrieves content from url.

        Args:
            url: input url

        Returns:
            data
        """

        # Local file
        if os.path.exists(url):
            with open(url, "rb") as f:
                return f.read()

        # Remote file
        with contextlib.closing(urlopen(Request(url, headers=self.headers))) as connection:
            return connection.read()
