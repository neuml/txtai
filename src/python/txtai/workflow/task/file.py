"""
FileTask module
"""

import os

from .base import Task


class FileTask(Task):
    """
    Task that processes file paths
    """

    def accept(self, element):
        # Only accept file paths that exist
        return super().accept(element) and isinstance(element, str) and os.path.exists(element)
