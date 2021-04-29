"""
FileTask module
"""

import re

from .url import UrlTask


class FileTask(UrlTask):
    """
    Task that processes file urls
    """

    def accept(self, element):
        # Only accept file URLs
        return super().accept(element) and re.match(UrlTask.FILE, element.lower())
