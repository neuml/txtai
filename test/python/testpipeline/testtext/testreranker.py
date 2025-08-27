"""
Reranker module tests
"""

import unittest

from txtai import Embeddings
from txtai.pipeline import Reranker, Similarity


class TestReranker(unittest.TestCase):
    """
    Reranker tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create single labels instance.
        """

        cls.data = [
            "US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day",
        ]

    def testRanker(self):
        """
        Test re-ranking pipeline
        """

        embeddings = Embeddings(content=True)
        embeddings.index(self.data)

        similarity = Similarity("neuml/colbert-bert-tiny", lateencode=True)

        ranker = Reranker(embeddings, similarity)
        self.assertEqual(ranker("lottery winner")[0]["id"], "4")
