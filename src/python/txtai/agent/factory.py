"""
Factory module
"""

from transformers.agents import CodeAgent, ReactCodeAgent, ReactJsonAgent

from .engine import LLMEngine
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

        constructor = ReactJsonAgent
        method = config.pop("method", None)
        if method == "code":
            constructor = CodeAgent
        elif method == "reactcode":
            constructor = ReactCodeAgent

        # Create LLMEngine
        llm = config.pop("llm")
        llm = LLMEngine(**llm) if isinstance(llm, dict) else LLMEngine(llm)

        # Create the agent process
        return constructor(tools=ToolFactory.create(config), llm_engine=llm, **config)
