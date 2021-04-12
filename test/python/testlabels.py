"""
Labels module tests
"""

import unittest

from txtai.pipeline import Labels, Similarity


class TestLabels(unittest.TestCase):
    """
    Labels tests
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

        cls.labels = Labels("prajjwal1/bert-medium-mnli")
        cls.similarity = Similarity(model=cls.labels)

    def testLabel(self):
        """
        Test labels with single text input
        """

        self.assertEqual(self.labels("This is the best sentence ever", ["positive", "negative"])[0][0], 0)

    def testLabelBatch(self):
        """
        Test labels with multiple text inputs
        """

        results = [l[0][0] for l in self.labels(["This is the best sentence ever", "This is terrible"], ["positive", "negative"])]
        self.assertEqual(results, [0, 1])

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

    def testSimilarityLong(self):
        """
        Test similarity with long text
        """

        uid = self.similarity("other", ["Very long text " * 1000, "other text"])[0][0]
        self.assertEqual(uid, 1)
