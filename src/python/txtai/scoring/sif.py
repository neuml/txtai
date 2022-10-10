"""
SIF module
"""

from .base import Scoring


class SIF(Scoring):
    """
    Smooth Inverse Frequency (SIF) scoring.
    """

    def __init__(self, config=None):
        super().__init__(config)

        # SIF configurable parameters
        self.a = self.config.get("a", 1e-3)

    def computefreq(self, tokens):
        # Default method computes frequency for a single entry
        # SIF uses word probabilities across entire index
        return self.wordfreq

    def score(self, freq, idf, length):
        # Calculate SIF score
        return self.a / (self.a + freq / self.tokens)
