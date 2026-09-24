"""
Texts module tests
"""

import unittest

from transformers import AutoTokenizer

from txtai.data import Texts


class TestTexts(unittest.TestCase):
    """
    Texts tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create tokenizer.
        """

        cls.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")

    def testPack(self):
        """
        Test packing rows into chunks up to maxlength
        """

        rows = ["a b c d", "e f g h", "i j k l"]
        length = len(self.tokenizer(rows[0])["input_ids"])

        # Two rows fill maxlength exactly and are packed into one chunk
        packed = Texts(self.tokenizer, None, length * 2, "pack").process({"text": list(rows)})
        self.assertEqual([len(chunk) for chunk in packed["input_ids"]], [length * 2, length])

        # Rows are never split across chunks
        packed = Texts(self.tokenizer, None, length * 2 - 1, "pack").process({"text": list(rows)})
        self.assertEqual([len(chunk) for chunk in packed["input_ids"]], [length] * 3)
