"""
Caption module tests
"""

import unittest

from PIL import Image

from transformers import AutoModelForImageTextToText, AutoImageProcessor, AutoTokenizer
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

        # Load passing models directly
        path = "ydshieh/vit-gpt2-coco-en"
        model = AutoModelForImageTextToText.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        processor = AutoImageProcessor.from_pretrained(path)

        caption = Caption((model, tokenizer, processor))
        self.assertEqual(caption(Image.open(Utils.PATH + "/books.jpg")), "a book shelf filled with books and a stack of books")
