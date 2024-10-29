"""
Labels module tests
"""

import unittest

from txtai.pipeline import Labels, Similarity


class TestLabels(unittest.TestCase):
    """
    Labels tests.
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

    def testLabel(self):
        """
        Test labels with single text input
        """

        self.assertEqual(self.labels("This is the best sentence ever", ["positive", "negative"])[0][0], 0)

    def testLabelFlatten(self):
        """
        Test labels with single text input, flattened to top text labels
        """

        self.assertEqual(self.labels("This is the best sentence ever", ["positive", "negative"], flatten=True)[0], "positive")

    def testLabelBatch(self):
        """
        Test labels with multiple text inputs
        """

        results = [l[0][0] for l in self.labels(["This is the best sentence ever", "This is terrible"], ["positive", "negative"])]
        self.assertEqual(results, [0, 1])

    def testLabelBatchFlatten(self):
        """
        Test labels with multiple text inputs, flattened to top text labels
        """

        results = [l[0] for l in self.labels(["This is the best sentence ever", "This is terrible"], ["positive", "negative"], flatten=True)]
        self.assertEqual(results, ["positive", "negative"])

    def testLabelFixed(self):
        """
        Test labels with a fixed label text classification model
        """

        labels = Labels(dynamic=False)

        # Get index of "POSITIVE" label
        index = labels.labels().index("POSITIVE")

        # Verify results
        self.assertEqual(labels("This is the best sentence ever")[0][0], index)
        self.assertEqual(labels("This is the best sentence ever", multilabel=True)[0][0], index)

    def testLabelFixedFlatten(self):
        """
        Test labels with a fixed label text classification model, flattened to top text labels
        """

        labels = Labels(dynamic=False)

        # Verify results
        self.assertEqual(labels("This is the best sentence ever", flatten=True)[0], "POSITIVE")
        self.assertEqual(labels("This is the best sentence ever", multilabel=True, flatten=True)[0], "POSITIVE")

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
