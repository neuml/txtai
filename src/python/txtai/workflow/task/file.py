"""
FileTask module
"""

import re

from .base import Task


class FileTask(Task):
    """
    Task that processes file urls
    """

    # File prefix
    FILE = r"file:\/\/"

    def accept(self, element):
        # Only accept file URLs
        return super().accept(element) and re.match(FileTask.FILE, element.lower())

    def prepare(self, element):
        return re.sub(FileTask.FILE, "", element)
