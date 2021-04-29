"""
UrlTask module
"""

import re

from .base import Task


class UrlTask(Task):
    """
    Task that processes urls
    """

    # URL prefix
    PREFIX = r"\w+:\/\/"

    # File prefix
    FILE = r"file:\/\/"

    def accept(self, element):
        # Only accept file URLs
        return super().accept(element) and re.match(UrlTask.PREFIX, element.lower())

    def prepare(self, element):
        return re.sub(UrlTask.FILE, "", element)
