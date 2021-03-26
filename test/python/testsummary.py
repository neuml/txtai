"""
Summary module tests
"""

import unittest

from txtai.pipeline import Summary


class TestSummary(unittest.TestCase):
    """
    Labels tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Create single summary instance
        """

        cls.text = (
            "Search is the base of many applications. Once data starts to pile up, users want to be able to find it. It’s the foundation "
            "of the internet and an ever-growing challenge that is never solved or done. The field of Natural Language Processing (NLP) is "
            "rapidly evolving with a number of new developments. Large-scale general language models are an exciting new capability "
            "allowing us to add amazing functionality quickly with limited compute and people. Innovation continues with new models "
            "and advancements coming in at what seems a weekly basis. This article introduces txtai, an AI-powered search engine "
            "that enables Natural Language Understanding (NLU) based search in any application."
        )

        cls.summary = Summary("sshleifer/distilbart-xsum-12-1")

    def testSummary(self):
        """
        Test summarization of text
        """

        self.assertEqual(self.summary(self.text, minlength=10, maxlength=10), "AI-powered search engine in the world")

    def testSummaryBatch(self):
        """
        Test batch summarization of text
        """

        summaries = self.summary([self.text, self.text], maxlength=10)
        self.assertEqual(len(summaries), 2)
