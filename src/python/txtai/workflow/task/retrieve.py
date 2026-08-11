"""
RetrieveTask module
"""

import os
import tempfile

from urllib.parse import urlparse

from ...pipeline import SafeOpen
from .base import Task


class RetrieveTask(Task):
    """
    Task that retrieves files and urls (local or remote) to a local directory.
    """

    def register(self, directory=None, flatten=True, headers=None, safeopen=False):
        """
        Adds retrieve parameters to task.

        Args:
            directory: local directory used to store retrieved files
            flatten: flatten input directory structure, defaults to True
            headers: http headers
            safeopen: if safe validation checks should be enabled
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
        self.safeinput = SafeOpen(headers, safeopen)
        self.safeoutput = SafeOpen(safeopen=directory if safeopen else safeopen, allowurl=False)

    def prepare(self, element):
        # Extract file path from URL
        path = urlparse(element).path

        # Validate input element
        url, _ = self.safeinput.valid(element)
        if url:
            if self.flatten:
                # Flatten directory structure (default)
                path = os.path.join(self.directory, os.path.basename(path))
                directory = None
            else:
                # Derive output path
                path = os.path.join(self.directory, os.path.normpath(path.lstrip("/")))
                directory = os.path.dirname(path)

            # Validate output path
            self.safeoutput.valid(path)

            # Create output directory, if necessary
            if directory:
                # Create local directory, if necessary
                os.makedirs(directory, exist_ok=True)

            # Retrieve data
            data = self.safeinput.retrieve(url)

            # Write to destination path
            with open(path, "wb") as output:
                output.write(data)

        # Return new file path
        return path
