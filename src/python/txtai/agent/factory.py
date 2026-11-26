"""
Factory module
"""

from .pydanticai import PydanticProcess
from .tool import ToolFactory


class ProcessFactory:
    """
    Methods to create agent processes.
    """

    @staticmethod
    def create(config):
        """
        Create an agent process runner. The agent process runner takes a list of tools and an LLM
        and executes an agent process flow.

        Args:
            config: agent configuration

        Returns:
            agent process runner
        """

        backend = config.pop("backend", "smolagents")
        if backend == "pydantic":
            return ProcessFactory.pydantic(config)

        if backend != "smolagents":
            raise ValueError(f"Unsupported agent backend: {backend}")

        return ProcessFactory.smolagents(config)

    @staticmethod
    def smolagents(config):
        """
        Create a smolagents process runner.

        Args:
            config: agent configuration

        Returns:
            agent process runner
        """

        # pylint: disable=C0415
        # smolagents is optional, only this backend requires it
        try:
            from smolagents import CodeAgent, ToolCallingAgent

            from .smolagents import PipelineModel, SmolagentsTool
        except ImportError as missing:
            raise ImportError('smolagents is not available - install "agent" extra to enable') from missing

        constructor = ToolCallingAgent
        method = config.pop("method", None)
        if method == "code":
            constructor = CodeAgent

        # Create model backed by LLM pipeline
        model = config.pop("model", config.pop("llm", None))
        model = PipelineModel(**model) if isinstance(model, dict) else PipelineModel(model)

        # Convert txtai tools to smolagents tools
        tools = [SmolagentsTool.create(tool) for tool in ToolFactory.create(config)]

        # Create the agent process
        return constructor(tools=tools, model=model, **config)

    @staticmethod
    def pydantic(config):
        """
        Create a Pydantic AI process runner.

        Args:
            config: agent configuration

        Returns:
            agent process runner
        """

        # LLM model path, configuration or instance
        model = config.pop("model", config.pop("llm", None))

        # Create the agent process
        return PydanticProcess(model, ToolFactory.create(config), **config)
