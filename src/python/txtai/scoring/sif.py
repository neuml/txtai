"""
SIF module
"""

from .base import Scoring


class SIF(Scoring):
    """
    Smooth Inverse Frequency (SIF) scoring.
    """

    def __init__(self, a=0.001):
        super().__init__()

        # SIF configurable parameters
        self.a = a

    def score(self, freq, idf, length):
        # Calculate SIF score
        return self.a / (self.a + freq / self.tokens)
