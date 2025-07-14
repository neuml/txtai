"""
Factory module
"""

from ...util import Resolver

from .ivfsparse import IVFSparse
from .pgsparse import PGSparse


class SparseANNFactory:
    """
    Methods to create Sparse ANN indexes.
    """

    @staticmethod
    def create(config):
        """
        Create an Sparse ANN.

        Args:
            config: index configuration parameters

        Returns:
            Sparse ANN
        """

        # ANN instance
        ann = None
        backend = config.get("backend", "ivfsparse")

        # Create ANN instance
        if backend == "ivfsparse":
            ann = IVFSparse(config)
        elif backend == "pgsparse":
            ann = PGSparse(config)
        else:
            ann = SparseANNFactory.resolve(backend, config)

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
            return Resolver()(backend)(config)
        except Exception as e:
            raise ImportError(f"Unable to resolve sparse ann backend: '{backend}'") from e
