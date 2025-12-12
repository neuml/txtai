"""
LLM module
"""

import logging

from .factory import GenerationFactory

from ..base import Pipeline

# Logging configuration
logger = logging.getLogger(__name__)


class LLM(Pipeline):
    """
    Pipeline for running large language models (LLMs). This class supports the following LLM backends:

      - Local LLMs with Hugging Face Transformers
      - Local LLMs with llama.cpp
      - Remote API LLMs with LiteLLM
      - Custom generation implementations
    """

    def __init__(self, path=None, method=None, **kwargs):
        """
        Creates a new LLM.

        Args:
            path: model path
            method: llm model framework, infers from path if not provided
            kwargs: model keyword arguments
        """

        # Default LLM if not provided
        path = path if path else "ibm-granite/granite-4.0-350m"

        # Generation instance
        self.generator = GenerationFactory.create(path, method, **kwargs)

    def __call__(self, text, maxlength=512, stream=False, stop=None, defaultrole="auto", stripthink=None, **kwargs):
        """
        Generates content. Supports the following input formats:

          - String or list of strings (instruction-tuned models must follow chat templates)
          - List of dictionaries with `role` and `content` key-values or lists of lists

        Args:
            text: text|list
            maxlength: maximum sequence length
            stream: stream response if True, defaults to False
            stop: list of stop strings, defaults to None
            defaultrole: default role to apply to text inputs (`auto` to infer (default), `user` for user chat messages or `prompt` for raw prompts)
            stripthink: strip thinking tags, defaults to False if stream is enabled, True otherwise
            kwargs: additional generation keyword arguments

        Returns:
            generated content
        """

        # Debug logging
        logger.debug(text)

        # Default stripthink to False when streaming, True otherwise
        stripthink = not stream if stripthink is None else stripthink

        # Run LLM generation
        return self.generator(text, maxlength, stream, stop, defaultrole, stripthink, **kwargs)

    def ischat(self):
        """
        Returns True if this LLM supports chat.

        Returns:
            True if this a chat model
        """

        return self.generator.ischat()

    def isvision(self):
        """
        Returns True if this LLM supports vision operations.

        Returns:
            True if this is a vision model
        """

        return self.generator.isvision()
