"""
HTMLToMarkdown module tests
"""

import unittest

from txtai.pipeline import HTMLToMarkdown


class TestHTMLToMarkdown(unittest.TestCase):
    """
    HTMLToMarkdown tests.
    """

    def testNestedListKeepsSiblingNumbers(self):
        """
        A nested item belongs to its own list. The parent list numbers only its direct items.
        """

        markdown = HTMLToMarkdown()

        nested = "<body><ol><li>one<ol><li>nested</li></ol></li><li>two</li></ol></body>"
        self.assertEqual(markdown(nested), "1. onenested\n2. two")

        flat = "<body><ol><li>one</li><li>two</li></ol></body>"
        self.assertEqual(markdown(flat), "1. one\n2. two")
