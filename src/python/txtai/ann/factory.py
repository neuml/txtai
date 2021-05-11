"""
Factory module
"""

from .annoy import Annoy
from .faiss import Faiss, FAISS
from .hnsw import HNSW


class ANNFactory:
    """
    Methods to create ANN models.
    """

    @staticmethod
    def create(config):
        """
        Create an ANN model.

        Args:
            config: index configuration parameters

        Returns:
            ANN
        """

        # ANN model
        model = None
        backend = config.get("backend")

        # Default backend if not provided, based on available libraries
        if not backend:
            backend = "faiss" if FAISS else "hnsw"

        # Create ANN instance
        if backend == "annoy":
            model = Annoy(config)
        elif backend == "hnsw":
            model = HNSW(config)
        else:
            # Raise error if trying to create a Faiss index without Faiss installed
            if not FAISS:
                raise ImportError("Faiss library is not installed")

            model = Faiss(config)

        # Store config back
        config["backend"] = backend

        return model
