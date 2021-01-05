"""
Labels module tests
"""

import unittest

import numpy as np

from txtai.pipeline import Labels, Similarity

class TestLabels(unittest.TestCase):
    """
    Labels tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Create single labels instance
        """

        cls.labels = Labels("prajjwal1/bert-medium-mnli")

    def testLabel(self):
        """
        Test labels
        """

        self.assertEqual(self.labels("This is the best sentence ever", ["positive", "negative"])[0][0], "positive")

    def testSimilarity(self):
        """
        Test similarity
        """

        similarity = Similarity(model=self.labels)

        data = ["US tops 5 million confirmed virus cases",
                "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
                "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
                "The National Park Service warns against sacrificing slower friends in a bear attack",
                "Maine man wins $1M from $25 lottery ticket",
                "Make huge profits without work, earn up to $100,000 a day"]

        uid = np.argmax(similarity("feel good story", data))
        self.assertEqual(data[uid], data[4])

        uid = np.argmax(similarity("feel good story", data[4]))
        self.assertEqual(uid, 0)
