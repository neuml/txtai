"""
Topic merging tests
"""

import unittest

from txtai.graph.topics import Topics


class TestTopicMerging(unittest.TestCase):
    """
    Tests for merging communities into topic labels.
    """

    def testLabelCollision(self):
        """
        Distinct term sets that generate the same label must retain all members.
        """

        topics = Topics({})
        entries = [(["a_b", "c"], [1, 2]), (["other"], [3, 4, 5]), (["a", "b_c"], [6, 7])]
        result = topics.merge(entries)
        self.assertEqual(list(result), ["a_b_c", "other"])
        self.assertEqual(result["a_b_c"], [1, 2, 6, 7])
        self.assertEqual(result["other"], [3, 4, 5])

    def testReorderedTerms(self):
        """
        Equal term sets still merge regardless of their order.
        """

        topics = Topics({})
        result = topics.merge([(["one", "two"], [1]), (["two", "one"], [2])])
        self.assertEqual(result, {"one_two": [1, 2]})
