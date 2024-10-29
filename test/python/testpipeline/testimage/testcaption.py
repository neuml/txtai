"""
Caption module tests
"""

import unittest

from PIL import Image

from txtai.pipeline import Caption

# pylint: disable = C0411
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
