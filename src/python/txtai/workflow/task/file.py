"""
FileTask module
"""

import os
import re

from .base import Task


class FileTask(Task):
    """
    Task that processes file paths
    """

    # File prefix
    FILE = r"file:\/\/"

    def accept(self, element):
        # Replace file prefixes
        element = re.sub(FileTask.FILE, "", element)

        # Only accept file paths that exist
        return super().accept(element) and isinstance(element, str) and os.path.exists(element)

    def prepare(self, element):
        # Replace file prefixes
        return re.sub(FileTask.FILE, "", element)
