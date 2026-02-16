"""
Caption module tests
"""

import unittest

from PIL import Image

from txtai.pipeline import Caption

# pylint: disable=C0411
from utils import Utils


class TestCaption(unittest.TestCase):
    """
    Caption tests.
    """

    def testCaption(self):
        """
        Test captions
        """

        caption = Caption()
        self.assertEqual(caption(Image.open(Utils.PATH + "/books.jpg")), "a book shelf filled with books and a stack of books")

    def testCaptionGenerator(self):
        """
        Test captions with generator input
        """

        caption = Caption()

        def image_gen():
            yield Utils.PATH + "/books.jpg"
            yield Utils.PATH + "/books.jpg"

        results = caption(image_gen())
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, str)
            self.assertTrue(len(r) > 0)

    def testCaptionIterator(self):
        """
        Test captions with iterator input
        """

        caption = Caption()
        results = caption(iter([Utils.PATH + "/books.jpg", Utils.PATH + "/books.jpg"]))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])
