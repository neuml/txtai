"""
BM25 module
"""

import math

from .base import Scoring


class BM25(Scoring):
    """
    BM25 scoring. Scores using Apache Lucene's version of BM25 which adds 1 to prevent
    negative scores.
    """

    def __init__(self, k1=0.1, b=0.75):
        super().__init__()

        # BM25 configurable parameters
        self.k1 = k1
        self.b = b

    def computeidf(self, freq):
        # Calculate BM25 IDF score
        return math.log(1 + (self.total - freq + 0.5) / (freq + 0.5))

    def score(self, freq, idf, length):
        # Calculate BM25 score
        k = self.k1 * ((1 - self.b) + self.b * length / self.avgdl)
        return idf * (freq * (self.k1 + 1)) / (freq + k)
