"""
Edit imports
"""

import difflib

from smolagents import Tool


class EditTool(Tool):
    """
    The EditTool modifies content in a file.
    """

    # pylint: disable=W0231
    def __init__(self):
        """
        Creates an EditTool.
        """

        # Tool parameters
        self.name = "edit"
        self.description = "Implementation of a file edit tool. Makes in-place edits to content in a file."
        self.inputs = {
            "path": {"type": "string", "description": "Path of file to edit"},
            "search": {"type": "string", "description": "String to replace"},
            "replace": {"type": "string", "description": "Replacement string"},
        }
        self.output_type = "any"

        # Validate parameters and initialize tool
        super().__init__()

    # pylint: disable=W0221
    def forward(self, path, search, replace):
        """
        Modifies content in a file. Returns a diff of the changes, if any.

        Args:
            path: file to edit
            search: string to replace
            replace: replacement

        Returns:
            diff
        """

        content = None
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Make edits
        modified = content.replace(search, replace)
        if modified != content:
            with open(path, "w", encoding="utf-8") as f:
                f.write(modified)

        # Return diff of file edits
        return "".join(difflib.unified_diff(content.splitlines(True), modified.splitlines(True), path, path))
