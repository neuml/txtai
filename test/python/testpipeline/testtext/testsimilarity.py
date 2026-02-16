"""
Similarity module tests
"""

import unittest

from txtai.pipeline import Similarity


class TestSimilarity(unittest.TestCase):
    """
    Similarity tests.
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

        cls.similarity = Similarity("prajjwal1/bert-medium-mnli")

    def testCrossEncoder(self):
        """
        Test cross-encoder similarity model
        """

        similarity = Similarity("cross-encoder/ms-marco-MiniLM-L-2-v2", crossencode=True)
        uid = similarity("Who won the lottery?", self.data)[0][0]
        self.assertEqual(self.data[uid], self.data[4])

    def testCrossEncoderBatch(self):
        """
        Test cross-encoder similarity model with multiple inputs
        """

        similarity = Similarity("cross-encoder/ms-marco-MiniLM-L-2-v2", crossencode=True)
        results = [r[0][0] for r in similarity(["Who won the lottery?", "Where did an iceberg collapse?"], self.data)]
        self.assertEqual(results, [4, 1])

    def testLateEncoder(self):
        """
        Test late-encoder similarity model
        """

        similarity = Similarity("neuml/pylate-bert-tiny", lateencode=True)
        uid = similarity("Who won the lottery?", self.data)[0][0]
        self.assertEqual(self.data[uid], self.data[4])

        # Test encode method
        # pylint: disable=E1101
        self.assertEqual(similarity.encode(["Who won the lottery?"], "data").shape, (1, 8, 128))

    def testLateEncoderBatch(self):
        """
        Test late-encoder similarity model with multiple inputs
        """

        similarity = Similarity("neuml/colbert-bert-tiny", lateencode=True)
        results = [r[0][0] for r in similarity(["Who won the lottery?", "Where did an iceberg collapse?"], self.data)]
        self.assertEqual(results, [4, 1])

    def testSimilarity(self):
        """
        Test similarity with single query
        """

        uid = self.similarity("feel good story", self.data)[0][0]
        self.assertEqual(self.data[uid], self.data[4])

    def testSimilarityBatch(self):
        """
        Test similarity with multiple queries
        """

        results = [r[0][0] for r in self.similarity(["feel good story", "climate change"], self.data)]
        self.assertEqual(results, [4, 1])

    def testSimilarityFixed(self):
        """
        Test similarity with a fixed label text classification model
        """

        similarity = Similarity(dynamic=False)

        # Test with query as label text and label id
        self.assertLessEqual(similarity("negative", ["This is the best sentence ever"])[0][1], 0.1)
        self.assertLessEqual(similarity("0", ["This is the best sentence ever"])[0][1], 0.1)

    def testSimilarityLong(self):
        """
        Test similarity with long text
        """

        uid = self.similarity("other", ["Very long text " * 1000, "other text"])[0][0]
        self.assertEqual(uid, 1)

    def testCrossEncoderLong(self):
        """
        Test cross-encoder with very long text (validates truncation=True)
        """

        similarity = Similarity("cross-encoder/ms-marco-MiniLM-L-2-v2", crossencode=True)

        # This should not raise an error thanks to truncation=True
        results = similarity("short query", ["Very long text " * 5000, "short answer"])
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0][1], float)
