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
        # Only process string elements
        if not isinstance(element, str):
            return False

        # Replace file prefixes
        element = re.sub(FileTask.FILE, "", element)

        # Only accept file paths that exist
        return super().accept(element) and os.path.exists(element)

    def prepare(self, element):
        # Replace file prefixes
        return re.sub(FileTask.FILE, "", element)
