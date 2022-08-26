"""
Factory module
"""

from .annoy import Annoy, ANNOY
from .faiss import Faiss
from .hnsw import HNSW, HNSWLIB
from .qdrant import Qdrant, QDRANT


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
        backend = config.get("backend", "faiss")

        # Create ANN instance
        if backend == "annoy":
            if not ANNOY:
                raise ImportError('annoy library is not available - install "similarity" extra to enable')

            model = Annoy(config)
        elif backend == "hnsw":
            if not HNSWLIB:
                raise ImportError('hnswlib library is not available - install "similarity" extra to enable')

            model = HNSW(config)
        elif backend == "qdrant":
            if not QDRANT:
                raise ImportError('qdrant-client library is not available - install "qdrant" extra to enable')
            model = Qdrant(config)
        else:
            model = Faiss(config)

        # Store config back
        config["backend"] = backend

        return model
