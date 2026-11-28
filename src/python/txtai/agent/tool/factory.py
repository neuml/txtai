"""
Factory module
"""

import inspect

from types import FunctionType, MethodType

from ...embeddings import Embeddings
from .base import Tool
from .bash import BashTool
from .edit import EditTool
from .embeddings import EmbeddingsTool
from .function import FunctionTool
from .glob import GlobTool
from .grep import GrepTool
from .question import QuestionTool
from .read import ReadTool
from .schema import ToolSchema, TypeHintError
from .skill import SkillTool
from .todo import TodoWriteTool
from .websearch import WebSearchTool
from .write import WriteTool


class ToolFactory:
    """
    Methods to create tools.
    """

    # Default toolkit, in name order
    DEFAULTS = {
        "bash": BashTool,
        "edit": EditTool,
        "glob": GlobTool,
        "grep": GrepTool,
        "question": QuestionTool,
        "read": ReadTool,
        "todowrite": TodoWriteTool,
        "websearch": WebSearchTool,
        "write": WriteTool,
    }

    # Backwards compatible mappings
    DEFAULTS["webview"] = DEFAULTS["read"]

    @staticmethod
    def create(config, defaults=None, mcp=None):
        """
        Creates a new list of tools. This method iterates of the `tools` configuration option and creates a Tool instance
        for each entry. This supports the following:

          - Tool instance
          - Dictionary with `name`, `description`, `inputs`, `output` and `target` function configuration
          - String with a tool alias name

        Args:
            config: agent configuration
            defaults: additional default tools provided by the agent backend, {alias name: tool class}
            mcp: function that loads the tools of an MCP server url, provided by the agent backend

        Returns:
            list of tools
        """

        # Default tools, in name order
        defaults = dict(sorted({**ToolFactory.DEFAULTS, **(defaults if defaults else {})}.items()))

        tools = []
        for tool in config.pop("tools", []):
            # Create tool from function and it's documentation
            if not ToolFactory.istool(tool) and (isinstance(tool, (FunctionType, MethodType)) or hasattr(tool, "__call__")):
                tool = ToolFactory.createtool(tool)

            # Create tool from input dictionary
            elif isinstance(tool, dict):
                # Get target function
                target = tool.get("target")

                # Create tool from input dictionary
                tool = (
                    EmbeddingsTool(tool)
                    if isinstance(target, Embeddings) or any(x in tool for x in ["container", "path"])
                    else ToolFactory.createtool(target, tool)
                )

            # Get default tool, if applicable
            elif isinstance(tool, str) and tool in defaults:
                tool = defaults[tool]()

            # Get ALL default tools, if applicable
            elif isinstance(tool, str) and tool == "defaults":
                tools.extend(default() for name, default in defaults.items() if name != "webview")
                tool = None

            # Support importing MCP tool collections
            elif isinstance(tool, str) and tool.startswith("http"):
                if not mcp:
                    raise ValueError("MCP tools are only supported by the smolagents backend")

                tools.extend(mcp(tool))
                tool = None

            # Load skill.md files
            elif isinstance(tool, str) and tool.endswith(".md"):
                tool = SkillTool(tool)

            elif isinstance(tool, str):
                raise ValueError(f"Unknown tool: {tool}")

            # Add tool
            if tool:
                tools.append(tool)

        return tools

    @staticmethod
    def createtool(target, config=None):
        """
        Creates a new Tool.

        Args:
            target: target object or function
            config: optional tool configuration

        Returns:
            Tool
        """

        config = config if config else {}

        try:
            # Try to create using type hints - this fails when no annotations are available
            return ToolFactory.fromhints(target, config)
        except TypeError:
            return ToolFactory.fromdocs(target, config)

    @staticmethod
    def istool(tool):
        """
        Checks if tool is a tool instance. This is a txtai tool or an object with the same tool attributes, such as a tool
        from an agent framework.

        Args:
            tool: object to check

        Returns:
            True if tool is a tool instance
        """

        return isinstance(tool, Tool) or all(hasattr(tool, name) for name in ("name", "description", "inputs", "output_type", "forward"))

    @staticmethod
    def default(name):
        """
        Creates a default tool by alias name.

        Args:
            name: default tool alias name

        Returns:
            Tool
        """

        return ToolFactory.DEFAULTS[name]()

    @staticmethod
    def fromhints(target, config):
        """
        Creates a tool from method type hints and documentation.

        Args:
            target: target function
            config: tool configuration

        Returns:
            Tool
        """

        schema = ToolFactory.unions(ToolSchema.create(target))

        # Functions with inputs must have a return type hint
        if "return" not in schema:
            if schema["parameters"]["properties"]:
                raise TypeHintError("Tool return type not found: make sure your function has a return type hint!")

            schema["return"] = {"type": "null"}

        # Return types that aren't a single type (i.e. Any or Union) are any
        output = schema["return"].get("type")
        output = output if isinstance(output, str) else "any"

        # Inputs without a type accept any value and optional inputs are nullable. Keys are ordered as smolagents orders them.
        inputs = {}
        for name, parameter in schema["parameters"]["properties"].items():
            description = parameter.pop("description", None)
            inputs[name] = {"type": parameter.pop("type", "any"), **parameter}

            if name not in schema["parameters"].get("required", []):
                inputs[name]["nullable"] = True

            if description is not None:
                inputs[name]["description"] = description

        # Create function tool
        return FunctionTool(
            {
                "name": config.get("name", schema["name"]),
                "description": config.get("description", schema["description"]),
                "inputs": inputs,
                "output": output,
                "target": target,
            }
        )

    @staticmethod
    def unions(schema):
        """
        Converts `anyOf` unions of basic types to a list of types, as smolagents does.

        Args:
            schema: JSON schema

        Returns:
            JSON schema
        """

        if isinstance(schema, list):
            return [ToolFactory.unions(x) for x in schema]

        if isinstance(schema, dict):
            schema = {key: ToolFactory.unions(value) for key, value in schema.items()}
            if "anyOf" in schema and all(isinstance(x.get("type"), str) for x in schema["anyOf"]):
                schema = {"type": sorted(x["type"] for x in schema.pop("anyOf")), **schema}

        return schema

    @staticmethod
    def fromdocs(target, config):
        """
        Creates a tool from method documentation.

        Args:
            target: target object or function
            config: tool configuration

        Returns:
            Tool
        """

        # Get function name and target - use target if it's a function or method, else use target.__call__
        name = target.__name__ if isinstance(target, (FunctionType, MethodType)) or not hasattr(target, "__call__") else target.__class__.__name__
        target = target if isinstance(target, (FunctionType, MethodType)) or not hasattr(target, "__call__") else target.__call__

        # Extract target documentation
        doc = inspect.getdoc(target)
        description, parameters, _ = ToolSchema.docstring(doc.strip()) if doc else (None, {}, None)

        # Get list of required parameters
        signature = inspect.signature(target)
        inputs = {}
        for pname, param in signature.parameters.items():
            if (
                param.default == inspect.Parameter.empty
                and pname in parameters
                and param.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
            ):
                inputs[pname] = {"type": "any", "description": parameters[pname]}

        # Create function tool
        return FunctionTool(
            {
                "name": config.get("name", name.lower()),
                "description": config.get("description", description),
                "inputs": config.get("inputs", inputs),
                "target": config.get("target", target),
            }
        )
