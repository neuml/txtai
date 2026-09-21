"""
Factory module
"""

from ...util import Resolver

from ..base import ANN
from .annoy import Annoy
from .faiss import Faiss, FAISS
from .ggml import GGML
from .hnsw import HNSW
from .milvus import Milvus
from .numpy import NumPy
from .pgvector import PGVector
from .rabitq import RabitQ
from .sqlite import SQLite
from .torch import Torch
from .turbovec import TurboVec
from .zvec import Zvec

BACKENDS = {
    "annoy": Annoy,
    "faiss": Faiss,
    "hnsw": HNSW,
    "milvus": Milvus,
    "ggml": GGML,
    "numpy": NumPy,
    "pgvector": PGVector,
    "sqlite": SQLite,
    "torch": Torch,
    "turbovec": TurboVec,
    "zvec": Zvec,
    "rabitq": RabitQ,
}


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
        backend = config.get("backend", "faiss" if FAISS else "numpy")

        # Create ANN instance
        if backend in BACKENDS:
            ann = BACKENDS[backend](config)
        else:
            ann = ANNFactory.resolve(backend, config)

        # Store config back
        config["backend"] = backend

        return ann

    @staticmethod
    def resolve(backend, config):
        """
        Attempt to resolve a custom backend.

        Args:
            backend: backend class
            config: index configuration parameters

        Returns:
            ANN
        """

        try:
            return Resolver()(backend, ANN)(config)
        except Exception as e:
            raise ImportError(f"Unable to resolve ann backend: '{backend}'") from e
