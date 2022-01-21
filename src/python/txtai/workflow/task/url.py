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

    def accept(self, element):
        # Only accept elements that start with a url prefix
        return super().accept(element) and re.match(UrlTask.PREFIX, element.lower())
