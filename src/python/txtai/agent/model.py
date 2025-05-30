"""
Model module
"""

import re

from enum import Enum

from smolagents import ChatMessage, Model, get_clean_message_list, tool_role_conversions
from smolagents.models import get_tool_call_from_text, remove_stop_sequences

from ..pipeline import LLM


class PipelineModel(Model):
    """
    Model backed by a LLM pipeline.
    """

    def __init__(self, path=None, method=None, **kwargs):
        """
        Creates a new LLM model.

        Args:
            path: model path or instance
            method: llm model framework, infers from path if not provided
            kwargs: model keyword arguments
        """

        self.llm = path if isinstance(path, LLM) else LLM(path, method, **kwargs)
        self.maxlength = 8192

        # Set base class parameters
        self.model_id = self.llm.generator.path

        # Call parent constructor
        super().__init__(flatten_messages_as_text=not self.llm.isvision(), **kwargs)

    # pylint: disable=W0613
    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        """
        Runs LLM inference. This method signature must match the smolagents specification.

        Args:
            messages: list of messages to run
            stop_sequences: optional list of stop sequences
            response_format: response format to use in the model's response.
            tools_to_call_from: list of tools that the model can use to generate responses.
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
            response = remove_stop_sequences(response, stop_sequences)

        # Load response into a chat message
        message = ChatMessage(role="assistant", content=response)

        # Extract first tool action, if necessary
        if tools_to_call_from:
            message.tool_calls = [
                get_tool_call_from_text(
                    re.sub(r".*?Action:(.*?\n\}).*", r"\1", response, flags=re.DOTALL), self.tool_name_key, self.tool_arguments_key
                )
            ]

        return message

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
        messages = get_clean_message_list(messages, role_conversions=tool_role_conversions, flatten_messages_as_text=self.flatten_messages_as_text)

        # Ensure all roles are strings and not enums for compability across LLM frameworks
        for message in messages:
            if "role" in message:
                message["role"] = message["role"].value if isinstance(message["role"], Enum) else message["role"]

        return messages
