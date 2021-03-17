"""
FileTask module
"""

import os
import re

from urllib.parse import urlparse

from .base import Task


class FileTask(Task):
    """
    Task that processes file urls
    """

    def accept(self, element):
        # Only accept file URLs
        return super().accept(element) and re.match(r"file:\/\/", element.lower())

    def prepare(self, element):
        # Transform file urls to local paths
        url = urlparse(element)
        return os.path.abspath(os.path.join(url.netloc, url.path))
