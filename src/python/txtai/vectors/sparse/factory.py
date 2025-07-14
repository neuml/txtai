"""
Factory module
"""

from ...util import Resolver

from .sbert import SparseSTVectors


class SparseVectorsFactory:
    """
    Methods to create sparse vector models.
    """

    @staticmethod
    def create(config, models=None):
        """
        Create a Vectors model instance.

        Args:
            config: vector configuration
            models: models cache

        Returns:
            Vectors
        """

        # Get vector method
        method = config.get("method", "sentence-transformers")

        # Sentence Transformers vectors
        if method == "sentence-transformers":
            return SparseSTVectors(config, None, models) if config and config.get("path") else None

        # Resolve custom method
        return SparseVectorsFactory.resolve(method, config, models) if method else None

    @staticmethod
    def resolve(backend, config, models):
        """
        Attempt to resolve a custom backend.

        Args:
            backend: backend class
            config: vector configuration
            models: models cache

        Returns:
            Vectors
        """

        try:
            return Resolver()(backend)(config, None, models)
        except Exception as e:
            raise ImportError(f"Unable to resolve sparse vectors backend: '{backend}'") from e
