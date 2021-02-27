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
    def create(method):
        """
        Factory method to construct a Scoring object.

        Args:
            method: scoring method (bm25, sif, tfidf)

        Returns:
            Scoring
        """

        if method == "bm25":
            return BM25()
        if method == "sif":
            return SIF()
        if method == "tfidf":
            # Default scoring object implements tf-idf
            return Scoring()

        return None
