"""
Function imports
"""

from smolagents import Tool


class FunctionTool(Tool):
    """
    Creates a FunctionTool. A FunctionTool takes descriptive configuration and injects it along with a target function
    into an LLM prompt.
    """

    # pylint: disable=W0231
    def __init__(self, config):
        """
        Creates a FunctionTool.

        Args:
            config: `name`, `description`, `inputs`, `output` and `target` configuration
        """

        # Tool parameters
        self.name = config["name"]
        self.description = config["description"]
        self.inputs = config["inputs"]
        self.output_type = config.get("output", config.get("output_type", "any"))
        self.target = config["target"]

        # pylint: disable=C0103
        # Skip forward signature validation
        self.skip_forward_signature_validation = True

        # Validate parameters and initialize tool
        super().__init__()

    def forward(self, *args, **kwargs):
        """
        Runs target function.

        Args:
            args: positional args
            kwargs: keyword args

        Returns:
            result
        """

        return self.target(*args, **kwargs)
