"""
FileTask module
"""

import os
import re

from urllib.request import url2pathname

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

        # Normalize file paths
        element = self.normalize(element)

        # Only accept file paths that exist
        return super().accept(element) and os.path.exists(element)

    def prepare(self, element):
        return self.normalize(element)

    def normalize(self, element):
        """
        Normalizes a local file path, decoding file URIs when necessary.

        Args:
            element: file path or URI

        Returns:
            local file path
        """

        # Decode file URIs while preserving literal percent sequences in plain paths.
        return url2pathname(re.sub(FileTask.FILE, "", element)) if element.startswith("file://") else element
