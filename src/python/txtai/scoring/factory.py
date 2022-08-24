"""
Factory module
"""

from .base import Scoring
from .bm25 import BM25
from .sif import SIF


class ScoringFactory:
    """
    Methods to create Scoring models.
    """

    @staticmethod
    def create(config):
        """
        Factory method to construct a Scoring object.

        Args:
            config: scoring configuration parameters - supports bm25, sif, tfidf

        Returns:
            Scoring
        """

        # Support string and dict configuration
        if isinstance(config, str):
            config = {"method": config}

        method = config.get("method") if config else None

        if method == "bm25":
            return BM25(config)
        if method == "sif":
            return SIF(config)
        if method == "tfidf":
            # Default scoring object implements tf-idf
            return Scoring(config)

        return None
