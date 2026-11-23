"""
Factory module
"""

from smolagents import CodeAgent, ToolCallingAgent

from .model import PipelineModel
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

        constructor = ToolCallingAgent
        method = config.pop("method", None)
        if method == "code":
            constructor = CodeAgent

        # Create model backed by LLM pipeline
        model = config.pop("model", config.pop("llm", None))
        model = PipelineModel(**model) if isinstance(model, dict) else PipelineModel(model)

        # Create the agent process
        return constructor(tools=ToolFactory.create(config), model=model, **config)

    @staticmethod
    def pydantic(config):
        """
        Create a Pydantic AI process runner.

        Args:
            config: agent configuration

        Returns:
            agent process runner
        """

        # Only import Pydantic AI when this backend is used
        try:
            # pylint: disable=C0415
            from .pydanticai import PydanticModel, PydanticProcess
        except ImportError as missing:
            raise ImportError('pydantic-ai-slim is not available - install "agent" extra to enable') from missing

        # Create model backed by LLM pipeline
        model = config.pop("model", config.pop("llm", None))
        model = PydanticModel(**model) if isinstance(model, dict) else PydanticModel(model)

        # Create the agent process
        return PydanticProcess(model, ToolFactory.create(config), **config)
