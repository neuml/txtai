"""
Factory module
"""

from .annoy import Annoy
from .faiss import Faiss
from .hnsw import HNSW


class ANNFactory:
    """
    Methods to create ANN indexes.
    """

    @staticmethod
    def create(config):
        """
        Create an ANN.

        Args:
            config: index configuration parameters

        Returns:
            ANN
        """

        # ANN instance
        ann = None
        backend = config.get("backend", "faiss")

        # Create ANN instance
        if backend == "annoy":
            ann = Annoy(config)
        elif backend == "hnsw":
            ann = HNSW(config)
        else:
            ann = Faiss(config)

        # Store config back
        config["backend"] = backend

        return ann
