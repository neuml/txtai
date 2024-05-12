"""
SIF module
"""

import numpy as np

from .tfidf import TFIDF


class SIF(TFIDF):
    """
    Smooth Inverse Frequency (SIF) scoring.
    """

    def __init__(self, config=None):
        super().__init__(config)

        # SIF configurable parameters
        self.a = self.config.get("a", 1e-3)

    def computefreq(self, tokens):
        # Default method computes frequency for a single entry
        # SIF uses word frequencies across entire index
        return {token: self.wordfreq[token] for token in tokens}

    def score(self, freq, idf, length):
        # Set freq to word frequencies across entire index when freq and idf shape don't match
        if isinstance(freq, np.ndarray) and freq.shape != idf.shape:
            freq.fill(freq.sum())

        # Calculate SIF score
        return self.a / (self.a + freq / self.tokens)
