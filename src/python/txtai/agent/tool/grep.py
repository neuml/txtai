"""
Grep imports
"""

import glob
import os
import re

from smolagents import Tool


class GrepTool(Tool):
    """
    The GrepTool "greps" for file matches.
    """

    # pylint: disable=W0231
    def __init__(self):
        """
        Creates a GrepTool.
        """

        # Tool parameters
        self.name = "grep"
        self.description = (
            "Implementation of a grep tool. Searches a list of files for matches. "
            "Returns a dictionary with the file path as the key and list of matches as the value."
        )
        self.inputs = {
            "search": {"type": "string", "description": "Search grep pattern"},
            "files": {"type": "string", "description": "File or file glob pattern to search"},
        }
        self.output_type = "any"

        # Validate parameters and initialize tool
        super().__init__()

    # pylint: disable=W0221
    def forward(self, search, files):
        """
        Greps a set of files for a pattern.

        Args:
            search: search pattern
            files: file or file glob pattern to search

        Returns:
            {path: [matches]}
        """

        # Compile the search pattern
        pattern = re.compile(search)

        # Format file pattern
        files = os.path.join(files, "*") if os.path.isdir(files) else files

        # Iterate over each found file
        results = {}
        for path in glob.glob(files, recursive=True):
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        # Search for matches line-by-line
                        matches = []
                        for line in f:
                            if pattern.search(line):
                                matches.append(line)

                        if matches:
                            results[path] = matches

                except (UnicodeDecodeError, FileNotFoundError):
                    pass

        return results
