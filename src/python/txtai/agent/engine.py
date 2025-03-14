"""
Engine module
"""

from enum import Enum

from transformers.agents import llm_engine

from ..pipeline import LLM


class LLMEngine:
    """
    Engine backed by a txtai LLM pipeline.
    """

    def __init__(self, path=None, method=None, **kwargs):
        """
        Creates a new LLM engine.

        Args:
            path: model path or instance
            method: llm model framework, infers from path if not provided
            kwargs: model keyword arguments
        """

        self.llm = path if isinstance(path, LLM) else LLM(path, method, **kwargs)
        self.maxlength = 8192

    def __call__(self, messages, stop_sequences=None, **kwargs):
        """
        Runs LLM inference. This method signature must match the Transformers Agents specification.

        Args:
            messages: list of messages to run
            stop_sequences: optional list of stop sequences
            kwargs: additional keyword arguments

        Returns:
            result
        """

        # Get clean message list
        messages = self.clean(messages)

        # Get LLM output
        response = self.llm(messages, maxlength=self.maxlength, stop=stop_sequences, **kwargs)

        # Remove stop sequences from LLM output
        if stop_sequences is not None:
            for stop in stop_sequences:
                if response[-len(stop) :] == stop:
                    response = response[: -len(stop)]

        return response

    def parameters(self, maxlength):
        """
        Set LLM inference parameters.

        Args:
            maxlength: maximum sequence length
        """

        self.maxlength = maxlength

    def clean(self, messages):
        """
        Gets a clean message list.

        Args:
            messages: input messages

        Returns:
            clean messages
        """

        # Get clean message list
        messages = llm_engine.get_clean_message_list(messages, role_conversions=llm_engine.llama_role_conversions)

        # Ensure all roles are strings and not enums for compability across LLM frameworks
        for message in messages:
            if "role" in message:
                message["role"] = message["role"].value if isinstance(message["role"], Enum) else message["role"]

        return messages
