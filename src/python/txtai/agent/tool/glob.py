"""
Glob imports
"""

import glob as g
import os

from smolagents import Tool


class GlobTool(Tool):
    """
    The GlobTool finds matching file patterns.
    """

    # pylint: disable=W0231
    def __init__(self):
        """
        Creates a GlobTool.
        """

        # Tool parameters
        self.name = "glob"
        self.description = "Implementation of a glob tool. Finds files that match glob patterns."
        self.inputs = {
            "files": {"type": "string", "description": "File or file glob pattern to search"},
        }
        self.output_type = "any"

        # Validate parameters and initialize tool
        super().__init__()

    # pylint: disable=W0221
    def forward(self, files):
        """
        Lists files matching a glob pattern

        Args:
            files: files glob pattern

        Returns:
            list of matching files
        """

        # Format file pattern
        files = os.path.join(files, "*") if os.path.isdir(files) else files

        # Iterate over each found file
        return g.glob(files, recursive=True)
