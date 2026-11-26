"""
Factory module
"""

import inspect

from types import FunctionType, MethodType

import mcpadapt.core

from transformers.utils import chat_template_utils

from ...embeddings import Embeddings
from .base import Tool
from .bash import BashTool
from .edit import EditTool
from .embeddings import EmbeddingsTool
from .function import FunctionTool
from .glob import GlobTool
from .grep import GrepTool
from .read import ReadTool
from .skill import SkillTool
from .todo import TodoWriteTool
from .write import WriteTool

# Conditional import
try:
    from mcpadapt.smolagents_adapter import SmolAgentsAdapter
    from smolagents import PythonInterpreterTool, Tool as SmolagentsTool, UserInputTool, WebSearchTool

    SMOLAGENTS = True
except ImportError:
    SMOLAGENTS = False


class ToolFactory:
    """
    Methods to create tools.
    """

    # Default toolkit
    DEFAULTS = {
        "bash": BashTool,
        "edit": EditTool,
        "glob": GlobTool,
        "grep": GrepTool,
        "read": ReadTool,
        "todowrite": TodoWriteTool,
        "write": WriteTool,
    }

    # Default tools provided by smolagents, only available when smolagents is installed. Defaults stay in name order.
    SMOLAGENTS_DEFAULTS = ["python", "question", "websearch"]
    if SMOLAGENTS:
        DEFAULTS = dict(sorted({**DEFAULTS, "python": PythonInterpreterTool, "question": UserInputTool, "websearch": WebSearchTool}.items()))

    # Backwards compatible mappings
    DEFAULTS["webview"] = DEFAULTS["read"]

    @staticmethod
    def create(config):
        """
        Creates a new list of tools. This method iterates of the `tools` configuration option and creates a Tool instance
        for each entry. This supports the following:

          - Tool instance
          - Dictionary with `name`, `description`, `inputs`, `output` and `target` function configuration
          - String with a tool alias name

        Returns:
            list of tools
        """

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
            elif isinstance(tool, str) and (tool in ToolFactory.DEFAULTS or tool in ToolFactory.SMOLAGENTS_DEFAULTS):
                tool = ToolFactory.default(tool)

            # Get ALL default tools, if applicable
            elif isinstance(tool, str) and tool == "defaults":
                tools.extend(ToolFactory.default(name) for name in ToolFactory.DEFAULTS if name != "webview")
                tool = None

            # Support importing MCP tool collections
            elif isinstance(tool, str) and tool.startswith("http"):
                if not SMOLAGENTS:
                    raise ImportError('MCP tools require smolagents - install "agent" extra to enable')

                tools.extend(mcpadapt.core.MCPAdapt({"url": tool}, SmolAgentsAdapter()).tools())
                tool = None

            # Load skill.md files
            elif isinstance(tool, str) and tool.endswith(".md"):
                tool = SkillTool(tool)

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
        except (chat_template_utils.TypeHintParsingException, TypeError):
            return ToolFactory.fromdocs(target, config)

    @staticmethod
    def istool(tool):
        """
        Checks if tool is a txtai or smolagents tool instance.

        Args:
            tool: object to check

        Returns:
            True if tool is a tool instance
        """

        return isinstance(tool, Tool) or (SMOLAGENTS and isinstance(tool, SmolagentsTool))

    @staticmethod
    def default(name):
        """
        Creates a default tool by alias name.

        Args:
            name: default tool alias name

        Returns:
            Tool
        """

        if name not in ToolFactory.DEFAULTS and name in ToolFactory.SMOLAGENTS_DEFAULTS:
            raise ImportError(f'The {name} tool requires smolagents - install "agent" extra to enable')

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

        schema = chat_template_utils.get_json_schema(target)["function"]

        # Functions with inputs must have a return type hint
        if "return" not in schema:
            if schema["parameters"]["properties"]:
                raise chat_template_utils.TypeHintParsingException("Tool return type not found: make sure your function has a return type hint!")

            schema["return"] = {"type": "null"}

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
                "output": schema["return"]["type"],
                "target": target,
            }
        )

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
        description, parameters, _ = chat_template_utils.parse_google_format_docstring(doc.strip()) if doc else (None, {}, None)

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
