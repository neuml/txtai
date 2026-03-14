"""
Bash imports
"""

import subprocess

from smolagents import Tool


class BashTool(Tool):
    """
    The BashTool runs a command through a subprocess. This tool only allows a small subset of commands.
    More can be added through configuration.
    """

    # pylint: disable=W0231
    def __init__(self, allowed=None):
        """
        Creates a BashTool.

        Args:
            allowed: list of allowed commands to run, has limited set of defaults
        """

        # Tool parameters
        self.name = "bash"
        self.description = "Implementation of a bash shell subprocess tool. Runs a shell command and returns the output."
        self.inputs = {
            "command": {"type": "array", "description": "Command to run. Follows Python subprocess.open pattern for command as a list of arguments."}
        }
        self.output_type = "any"

        # Default list of allowed commands
        self.allowed = allowed if allowed else ["cat", "cut", "diff", "find", "grep", "head", "ls", "tail"]

        # Validate parameters and initialize tool
        super().__init__()

    # pylint: disable=W0221
    def forward(self, command):
        """
        Runs a shell command as a subprocess.

        Args:
            command: command arguments as a list

        Returns:
            command output
        """

        output = None
        if command and command[0] in self.allowed:
            output = subprocess.run(command, capture_output=True, text=True, check=False).stdout

        return output
