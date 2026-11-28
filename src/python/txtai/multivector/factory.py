"""
Factory module
"""

from ..util import Resolver

from .base import MultiVector
from .numpy import NumPy


class MultiVectorFactory:
    """
    Methods to create multi-vector components.
    """

    @staticmethod
    def create(embeddings, config):
        """
        Creates a multi-vector component.

        Args:
            embeddings: owning embeddings instance
            config: multi-vector configuration

        Returns:
            MultiVector
        """

        backend = config.get("backend", "numpy")
        multivector = NumPy(embeddings, config) if backend == "numpy" else MultiVectorFactory.resolve(backend, embeddings, config)

        # Store config back
        config["backend"] = backend
        return multivector

    @staticmethod
    def resolve(backend, embeddings, config):
        """
        Attempts to resolve a custom backend.

        Args:
            backend: backend class
            embeddings: owning embeddings instance
            config: multi-vector configuration

        Returns:
            MultiVector
        """

        try:
            return Resolver()(backend, MultiVector)(embeddings, config)
        except Exception as e:
            raise ImportError(f"Unable to resolve multivector backend: '{backend}'") from e
