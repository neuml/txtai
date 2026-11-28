"""
Factory module
"""

from ..util import Resolver

from .pydanticai import PydanticProcess
from .smolagents import SmolagentsProcess
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

        # Get agent backend
        backend = config.pop("backend", "smolagents")

        # smolagents converts tools to its own format
        if backend == "smolagents":
            return SmolagentsProcess.create(config)

        # LLM model path, configuration or instance
        model = config.pop("model", config.pop("llm", None))

        # txtai tools
        tools = ToolFactory.create(config)

        # Pydantic AI
        if backend == "pydantic":
            return PydanticProcess(model, tools, **config)

        # Resolve custom backend
        return ProcessFactory.resolve(backend, model, tools, config)

    @staticmethod
    def resolve(backend, model, tools, config):
        """
        Attempt to resolve a custom backend.

        Args:
            backend: backend class
            model: LLM model path, configuration or instance
            tools: list of txtai tools
            config: agent configuration

        Returns:
            agent process runner
        """

        try:
            return Resolver()(backend)(model, tools, **config)
        except Exception as e:
            raise ImportError(f"Unable to resolve agent backend: '{backend}'") from e
