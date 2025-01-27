"""
Factory module
"""

from ..util import Resolver

from .external import External
from .huggingface import HFVectors
from .litellm import LiteLLM
from .llama import LlamaCpp
from .m2v import Model2Vec
from .sbert import STVectors
from .words import WordVectors


class VectorsFactory:
    """
    Methods to create Vectors models.
    """

    @staticmethod
    def create(config, scoring=None, models=None):
        """
        Create a Vectors model instance.

        Args:
            config: vector configuration
            scoring: scoring instance
            models: models cache

        Returns:
            Vectors
        """

        # Determine vector method
        method = VectorsFactory.method(config)

        # External vectors
        if method == "external":
            return External(config, scoring, models)

        # LiteLLM vectors
        if method == "litellm":
            return LiteLLM(config, scoring, models)

        # llama.cpp vectors
        if method == "llama.cpp":
            return LlamaCpp(config, scoring, models)

        # Model2vec vectors
        if method == "model2vec":
            return Model2Vec(config, scoring, models)

        # Sentence Transformers vectors
        if method == "sentence-transformers":
            return STVectors(config, scoring, models) if config and config.get("path") else None

        # Word vectors
        if method == "words":
            return WordVectors(config, scoring, models)

        # Transformers vectors
        if HFVectors.ismethod(method):
            return HFVectors(config, scoring, models) if config and config.get("path") else None

        # Resolve custom method
        return VectorsFactory.resolve(method, config, scoring, models) if method else None

    @staticmethod
    def method(config):
        """
        Get or derive the vector method.

        Args:
            config: vector configuration

        Returns:
            vector method
        """

        # Determine vector method
        method = config.get("method")
        path = config.get("path")

        # Infer method from path, if blank
        if not method:
            if path:
                if LiteLLM.ismodel(path):
                    method = "litellm"
                elif LlamaCpp.ismodel(path):
                    method = "llama.cpp"
                elif Model2Vec.ismodel(path):
                    method = "model2vec"
                elif WordVectors.ismodel(path):
                    method = "words"
                else:
                    method = "transformers"
            elif config.get("transform"):
                method = "external"

        return method

    @staticmethod
    def resolve(backend, config, scoring, models):
        """
        Attempt to resolve a custom backend.

        Args:
            backend: backend class
            config: vector configuration
            scoring: scoring instance
            models: models cache

        Returns:
            Vectors
        """

        try:
            return Resolver()(backend)(config, scoring, models)
        except Exception as e:
            raise ImportError(f"Unable to resolve vectors backend: '{backend}'") from e
