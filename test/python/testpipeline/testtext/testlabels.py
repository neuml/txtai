"""
Labels module tests
"""

import unittest

from txtai.pipeline import Labels


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
