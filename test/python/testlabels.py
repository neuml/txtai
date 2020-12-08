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
        Tests labeling
        """

        labels = Labels()
        self.assertIsNotNone(labels("This is a test", "test"))
