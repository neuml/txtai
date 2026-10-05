"""
Read imports
"""

import re

from smolagents import Tool

from ...pipeline import Textractor


class ReadTool(Tool):
    """
    The ReadTool retrieves file or url content. This tool automatically extracts text content from
    binary files using the Textractor pipeline.
    """

    # pylint: disable=W0231
    def __init__(self, maxlength=40000):
        """
        Creates a ReadTool.

        Args:
            maxlength: Truncate content above this maxlength
        """

        # Tool parameters
        self.name = "read"
        self.description = (
            "Implementation of a file read tool. Returns file content. Also supports reading web content. "
            "Use this tool to browse webpages in addition to reading files."
        )
        self.inputs = {"path": {"type": "string", "description": "File path or url"}}
        self.output_type = "any"

        # Create textractor instance
        self.textractor = Textractor()
        self.maxlength = maxlength

        # Validate parameters and initialize tool
        super().__init__()

    # pylint: disable=W0221
    def forward(self, path):
        """
        Reads content from path.

        Args:
            path: file path or url

        Returns:
            content
        """

        content = self.textractor(path)
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Truncate content to max length
        return content[: self.maxlength]
