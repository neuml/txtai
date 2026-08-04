"""
Factory module
"""

import inspect

from types import FunctionType, MethodType

import mcpadapt.core

from mcpadapt.smolagents_adapter import SmolAgentsAdapter
from smolagents import PythonInterpreterTool, Tool, tool as CreateTool, UserInputTool, WebSearchTool
from transformers.utils import chat_template_utils, TypeHintParsingException

from ...embeddings import Embeddings
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


class ToolFactory:
    """
    Methods to create tools.
    """

    # Default toolkit. Stores constructors, not instances - these are built on first use by `default`.
    #
    # Instances must not be created here. A class body runs at import time, so a tool that needs an
    # optional dependency would raise while `txtai.agent` is being imported. That ImportError is
    # caught in agent/__init__.py, which then installs the placeholder Agent and reports that
    # smolagents is missing - regardless of which dependency was actually missing. ReadTool builds a
    # Textractor, which needs the "pipeline" extra, so installing only the documented "agent" extra
    # disabled agents entirely and pointed at the wrong package.
    DEFAULTS = {
        "bash": BashTool,
        "edit": EditTool,
        "glob": GlobTool,
        "grep": GrepTool,
        "python": PythonInterpreterTool,
        "question": UserInputTool,
        "read": ReadTool,
        "todowrite": TodoWriteTool,
        "websearch": WebSearchTool,
        "write": WriteTool,
    }

    # Backwards compatible mappings
    DEFAULTS["webview"] = DEFAULTS["read"]

    # Cache of default tool instances, keyed by constructor
    INSTANCES = {}

    @staticmethod
    def default(name):
        """
        Gets a default tool by alias name, creating it on first use.

        Instances are cached, so an alias and its backwards compatible mapping share a single tool,
        as do repeated agents in the same process.

        Args:
            name: default tool alias name

        Returns:
            Tool
        """

        constructor = ToolFactory.DEFAULTS[name]
        if constructor not in ToolFactory.INSTANCES:
            ToolFactory.INSTANCES[constructor] = constructor()

        return ToolFactory.INSTANCES[constructor]

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
            if not isinstance(tool, Tool) and (isinstance(tool, (FunctionType, MethodType)) or hasattr(tool, "__call__")):
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
            elif isinstance(tool, str) and tool in ToolFactory.DEFAULTS:
                tool = ToolFactory.default(tool)

            # Get ALL default tools, if applicable
            elif isinstance(tool, str) and tool == "defaults":
                tools.extend({ToolFactory.default(name) for name in ToolFactory.DEFAULTS})
                tool = None

            # Support importing MCP tool collections
            elif isinstance(tool, str) and tool.startswith("http"):
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

        try:
            # Try to create using CreateTool function - this fails when no annotations are available
            return CreateTool(target)
        except (TypeHintParsingException, TypeError):
            return ToolFactory.fromdocs(target, config if config else {})

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
