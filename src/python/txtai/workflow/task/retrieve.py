"""
RetrieveTask module
"""

import hashlib
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
        # Tracks output names already used in this run
        self.names = {}

    def prepare(self, element):
        # Validate input element
        url, exists = self.safeinput.valid(element)

        # Preserve local filenames; only parse remote inputs as URLs
        path = url if exists else urlparse(element).path
        if url:
            if self.flatten:
                # Flatten directory structure (default)
                path = self.uniquename(os.path.basename(path), element)
                directory = None
            else:
                # Derive output path
                path = os.path.join(self.directory, os.path.normpath(path.lstrip("/")))
                path = self.uniquename(os.path.relpath(path, self.directory), element)
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

    def uniquename(self, name, element):
        """
        Resolves a file name to a unique output path. The first input keeps the plain name.
        A later input mapping to the same name gets a short hash of the input element
        appended, which keeps the output deterministic and preserves all retrieved data.

        Args:
            name: file name relative to the output directory
            element: input element the name is derived from

        Returns:
            unique output path
        """

        # Fallback for inputs with no file name, e.g. urls ending in /
        if not name or name == ".":
            name = "download"

        # Deduplicate name across inputs
        if name in self.names:
            # Repeated input keeps its previously resolved name
            if self.names[name] != element:
                name = f"{name}-{hashlib.sha1(element.encode()).hexdigest()[:8]}"
        else:
            self.names[name] = element

        return os.path.join(self.directory, name)
