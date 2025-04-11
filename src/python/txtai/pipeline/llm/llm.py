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
        path = path if path else "google/flan-t5-base"

        # Generation instance
        self.generator = GenerationFactory.create(path, method, **kwargs)

    def __call__(self, text, maxlength=512, stream=False, stop=None, defaultrole="prompt", **kwargs):
        """
        Generates text. Supports the following input formats:

          - String or list of strings (instruction-tuned models must follow chat templates)
          - List of dictionaries with `role` and `content` key-values or lists of lists

        Args:
            text: text|list
            maxlength: maximum sequence length
            stream: stream response if True, defaults to False
            stop: list of stop strings, defaults to None
            defaultrole: default role to apply to text inputs (prompt for raw prompts (default) or user for user chat messages)
            kwargs: additional generation keyword arguments

        Returns:
            generated text
        """

        # Debug logging
        logger.debug(text)

        # Run LLM generation
        return self.generator(text, maxlength, stream, stop, defaultrole, **kwargs)

    def isvision(self):
        """
        Returns True if this LLM supports vision operations.

        Returns:
            True if this is a vision model
        """

        return self.generator.isvision()
