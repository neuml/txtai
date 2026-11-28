"""
Tool module
"""

from smolagents import PythonInterpreterTool, Tool


class SmolagentsTool(Tool):
    """
    Converts txtai agent tools to smolagents tools.
    """

    # Inputs are passed through to the txtai tool, which validates them
    skip_forward_signature_validation = True

    # Default tools only available with smolagents
    DEFAULTS = {"python": PythonInterpreterTool}

    # pylint: disable=W0231
    def __init__(self, tool):
        """
        Creates a new SmolagentsTool.

        Args:
            tool: txtai agent tool
        """

        self.tool = tool

        # Tool parameters
        self.name = tool.name
        self.description = tool.description
        self.inputs = tool.inputs
        self.output_type = tool.output

        # Validate parameters and initialize tool
        super().__init__()

    def __getattr__(self, name):
        """
        Reads attributes not set on this tool from the txtai tool (i.e. BashTool.allowed).

        Args:
            name: attribute name

        Returns:
            attribute value
        """

        # Not set while the object is created or copied
        if name == "tool":
            raise AttributeError(name)

        return getattr(self.tool, name)

    # pylint: disable=W0221
    def forward(self, *args, **kwargs):
        """
        Runs the txtai tool.

        Args:
            args: positional arguments
            kwargs: keyword arguments

        Returns:
            tool output
        """

        return self.tool(*args, **kwargs)

    @staticmethod
    def create(tool):
        """
        Creates a smolagents tool from a txtai agent tool. smolagents tools are returned unchanged.

        Args:
            tool: txtai agent tool or smolagents tool

        Returns:
            smolagents tool
        """

        return tool if isinstance(tool, Tool) else SmolagentsTool(tool)

    @staticmethod
    def mcp(url):
        """
        Loads the tools of an MCP server.

        Args:
            url: MCP server url

        Returns:
            list of smolagents tools
        """

        # pylint: disable=C0415
        import mcpadapt.core

        from mcpadapt.smolagents_adapter import SmolAgentsAdapter

        return mcpadapt.core.MCPAdapt({"url": url}, SmolAgentsAdapter()).tools()
