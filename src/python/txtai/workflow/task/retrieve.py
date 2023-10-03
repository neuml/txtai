"""
RetrieveTask module
"""

import os
import tempfile

from urllib.request import urlretrieve
from urllib.parse import urlparse

from .url import UrlTask


class RetrieveTask(UrlTask):
    """
    Task that retrieves urls (local or remote) to a local directory.
    """

    def register(self, directory=None, flatten=True):
        """
        Adds retrieve parameters to task.

        Args:
            directory: local directory used to store retrieved files
            flatten: flatten input directory structure, defaults to True
        """

        # pylint: disable=W0201
        # Create default temporary directory if not specified
        if not directory:
            # Save tempdir to prevent content from being deleted until this task is out of scope
            # pylint: disable=R1732
            self.tempdir = tempfile.TemporaryDirectory()
            directory = self.tempdir.name

        # Create output directory if necessary
        os.makedirs(directory, exist_ok=True)

        self.directory = directory
        self.flatten = flatten

    def prepare(self, element):
        # Extract file path from URL
        path = urlparse(element).path

        if self.flatten:
            # Flatten directory structure (default)
            path = os.path.join(self.directory, os.path.basename(path))
        else:
            # Derive output path
            path = os.path.join(self.directory, os.path.normpath(path.lstrip("/")))
            directory = os.path.dirname(path)

            # Create local directory, if necessary
            os.makedirs(directory, exist_ok=True)

        # Retrieve URL
        urlretrieve(element, path)

        # Return new file path
        return path
