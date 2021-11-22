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

    def __init__(self, action=None, select=None, unpack=True, column=None, merge="hstack", initialize=None, finalize=None, directory=None):
        super().__init__(action, select, unpack, column, merge, initialize, finalize)

        # Create default temporary directory if not specified
        if not directory:
            # Save tempdir to prevent content from being deleted until this task is out of scope
            self.tempdir = tempfile.TemporaryDirectory()
            directory = self.tempdir.name

        # Create output directory if necessary
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.directory = directory

    def prepare(self, element):
        # Run super() method to format element
        element = super().prepare(element)

        # Extract file name
        _, name = os.path.split(element)

        # Derive output path
        path = os.path.join(self.directory, name)

        # Retrieve URL
        urlretrieve(element, os.path.join(self.directory, name))

        # Return new file path
        return path
