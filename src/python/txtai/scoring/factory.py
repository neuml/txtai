"""
Factory module
"""

from ..util import Resolver

from .bm25 import BM25
from .sif import SIF
from .pgtext import PGText
from .tfidf import TFIDF


class ScoringFactory:
    """
    Methods to create Scoring indexes.
    """

    @staticmethod
    def create(config):
        """
        Factory method to construct a Scoring instance.

        Args:
            config: scoring configuration parameters - supports bm25, sif, tfidf

        Returns:
            Scoring
        """

        # Scoring instance
        scoring = None

        # Support string and dict configuration
        if isinstance(config, str):
            config = {"method": config}

        # Get scoring method
        method = config.get("method", "bm25")

        if method == "bm25":
            scoring = BM25(config)
        elif method == "sif":
            scoring = SIF(config)
        elif method == "pgtext":
            scoring = PGText(config)
        elif method == "tfidf":
            scoring = TFIDF(config)
        else:
            # Resolve custom method
            scoring = ScoringFactory.resolve(method, config)

        # Store config back
        config["method"] = method

        return scoring

    @staticmethod
    def resolve(backend, config):
        """
        Attempt to resolve a custom backend.

        Args:
            backend: backend class
            config: index configuration parameters

        Returns:
            Scoring
        """

        try:
            return Resolver()(backend)(config)
        except Exception as e:
            raise ImportError(f"Unable to resolve scoring backend: '{backend}'") from e
