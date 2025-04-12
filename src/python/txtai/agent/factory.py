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

        constructor = ToolCallingAgent
        method = config.pop("method", None)
        if method == "code":
            constructor = CodeAgent

        # Create model backed by LLM pipeline
        model = config.pop("model", config.pop("llm", None))
        model = PipelineModel(**model) if isinstance(model, dict) else PipelineModel(model)

        # Create the agent process
        return constructor(tools=ToolFactory.create(config), model=model, **config)
