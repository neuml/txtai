"""
Bash imports
"""

import subprocess

from smolagents import Tool

# Default maximum time in seconds to wait for a command to complete
TIMEOUT = 30


class BashTool(Tool):
    """
    The BashTool runs a command through a subprocess. This tool only allows a small subset of commands.
    More can be added through configuration.
    """

    # pylint: disable=W0231
    def __init__(self, allowed=None, timeout=None):
        """
        Creates a BashTool.

        Args:
            allowed: list of allowed commands to run, has limited set of defaults which are a best effort not a sandbox
            timeout: maximum number of seconds to wait for a command to complete, defaults to 30
        """

        # Tool parameters
        self.name = "bash"
        self.description = "Implementation of a bash shell subprocess tool. Runs a shell command and returns the output."
        self.inputs = {
            "command": {"type": "array", "description": "Command to run. Follows Python subprocess.open pattern for command as a list of arguments."}
        }
        self.output_type = "any"

        # Default list of allowed commands
        self.allowed = allowed if allowed else ["cat", "cut", "diff", "grep", "head", "ls", "tail"]

        # Maximum time to wait for a command
        self.timeout = timeout if timeout else TIMEOUT

        # Validate parameters and initialize tool
        super().__init__()

    # pylint: disable=W0221
    def forward(self, command):
        """
        Runs a shell command as a subprocess.

        stdin is closed and a timeout is applied. Otherwise an allowed command called with no file arguments
        (i.e. `cat`) reads from stdin and blocks the agent indefinitely.

        Args:
            command: command arguments as a list

        Returns:
            command output
        """

        output = None
        if command and command[0] in self.allowed:
            try:
                output = subprocess.run(command, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL, timeout=self.timeout).stdout
            except subprocess.TimeoutExpired:
                output = f"Command timed out after {self.timeout} seconds"

        return output
