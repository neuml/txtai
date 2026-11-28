"""
Process module
"""

# Conditional import
try:
    from smolagents import CodeAgent, ToolCallingAgent

    from .model import PipelineModel
    from .tool import SmolagentsTool

    SMOLAGENTS = True
except ImportError:
    SMOLAGENTS = False

from ..tool import ToolFactory


class SmolagentsProcess:
    """
    Creates smolagents process runners.
    """

    @staticmethod
    def create(config):
        """
        Create a smolagents process runner.

        Args:
            config: agent configuration

        Returns:
            agent process runner
        """

        if not SMOLAGENTS:
            raise ImportError('smolagents is not available - install "agent" extra to enable')

        constructor = ToolCallingAgent
        method = config.pop("method", None)
        if method == "code":
            constructor = CodeAgent

        # Create model backed by LLM pipeline
        model = config.pop("model", config.pop("llm", None))
        model = PipelineModel(**model) if isinstance(model, dict) else PipelineModel(model)

        # Convert txtai tools to smolagents tools
        tools = [SmolagentsTool.create(tool) for tool in ToolFactory.create(config, SmolagentsTool.DEFAULTS, SmolagentsTool.mcp)]

        # Create the agent process
        return constructor(tools=tools, model=model, **config)
