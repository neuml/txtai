"""
Factory module
"""

from ...util import Resolver

from .huggingface import HFGeneration
from .litellm import LiteLLM
from .llama import LlamaCpp


class GenerationFactory:
    """
    Methods to create generative models.
    """

    @staticmethod
    def create(path, method, **kwargs):
        """
        Creates a new Generation instance.

        Args:
            path: model path
            method: llm framework
            kwargs: model keyword arguments
        """

        # Derive method
        method = GenerationFactory.method(path, method)

        # LiteLLM generation
        if method == "litellm":
            return LiteLLM(path, **kwargs)

        # llama.cpp generation
        if method == "llama.cpp":
            return LlamaCpp(path, **kwargs)

        # Hugging Face Transformers generation
        if method == "transformers":
            return HFGeneration(path, **kwargs)

        # Resolve custom method
        return GenerationFactory.resolve(path, method, **kwargs)

    @staticmethod
    def method(path, method):
        """
        Get or derives the LLM framework.

        Args:
            path: model path
            method: llm framework

        Return:
            llm framework
        """

        if not method:
            if LiteLLM.ismodel(path):
                method = "litellm"
            elif LlamaCpp.ismodel(path):
                method = "llama.cpp"
            else:
                method = "transformers"

        return method

    @staticmethod
    def resolve(path, method, **kwargs):
        """
        Attempt to resolve a custom LLM framework.

        Args:
            path: model path
            method: llm framework
            kwargs: model keyword arguments

        Returns:
            Generation instance
        """

        try:
            return Resolver()(method)(path, **kwargs)
        except Exception as e:
            raise ImportError(f"Unable to resolve generation framework: '{method}'") from e
