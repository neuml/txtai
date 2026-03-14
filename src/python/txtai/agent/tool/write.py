"""
Write imports
"""

from smolagents import Tool


class WriteTool(Tool):
    """
    The WriteTool writes file content.
    """

    # pylint: disable=W0231
    def __init__(self):
        """
        Creates a WriteTool.
        """

        # Tool parameters
        self.name = "write"
        self.description = "Implementation of a file write tool. Writes content to file."
        self.inputs = {
            "path": {"type": "string", "description": "Output file path"},
            "content": {"type": "string", "description": "File content to write"},
        }
        self.output_type = "any"

        # Validate parameters and initialize tool
        super().__init__()

    # pylint: disable=W0221
    def forward(self, path, content):
        """
        Writes content to path.

        Args:
            path: output file path
            content: content to write
        """

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
