"""
BM25 module
"""

import numpy as np

from .tfidf import TFIDF


class BM25(TFIDF):
    """
    Best matching (BM25) scoring.
    """

    def __init__(self, config=None):
        super().__init__(config)

        # BM25 configurable parameters
        self.k1 = self.config.get("k1", 1.2)
        self.b = self.config.get("b", 0.75)

    def computeidf(self, freq):
        # Calculate BM25 IDF score
        return np.log(1 + (self.total - freq + 0.5) / (freq + 0.5))

    def score(self, freq, idf, length):
        # Calculate BM25 score
        k = self.k1 * ((1 - self.b) + self.b * length / self.avgdl)
        return idf * (freq * (self.k1 + 1)) / (freq + k)
