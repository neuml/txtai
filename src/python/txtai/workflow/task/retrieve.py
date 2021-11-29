"""
RetrieveTask module
"""

import os
import tempfile

from urllib.request import urlretrieve

from .url import UrlTask


class RetrieveTask(UrlTask):
    """
    Task that retrieves urls (local or remote) to a local directory.
    """

    def register(self, directory=None):
        """
        Adds retrieve parameters to task.

        Args:
            directory: local directory used to store retrieved files
        """

        # pylint: disable=W0201
        # Create default temporary directory if not specified
        if not directory:
            # Save tempdir to prevent content from being deleted until this task is out of scope
            # pylint: disable=R1732
            self.tempdir = tempfile.TemporaryDirectory()
            directory = self.tempdir.name

        # Create output directory if necessary
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.directory = directory

    def prepare(self, element):
        # Extract file name
        _, name = os.path.split(element)

        # Derive output path
        path = os.path.join(self.directory, name)

        # Retrieve URL
        urlretrieve(element, os.path.join(self.directory, name))

        # Return new file path
        return path
