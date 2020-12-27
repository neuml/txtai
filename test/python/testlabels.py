"""
Labels module tests
"""

import unittest

from txtai.pipeline import Labels

class TestLabels(unittest.TestCase):
    """
    Labels tests
    """

    def testLabel(self):
        """
        Test labels
        """

        labels = Labels()
        self.assertEqual(labels("This is the best sentence ever", ["positive", "negative"])[0][0], "positive")
