"""
Questions module tests
"""

import unittest

from txtai.pipeline import Questions


class TestQuestions(unittest.TestCase):
    """
    Questions tests.
    """

    def testNoAnswer(self):
        """
        Test unmapped span tokens alongside an ordinary context span
        """

        questions = Questions("distilbert-base-cased-distilled-squad", gpu=False)
        context = "Paris is the capital of France."
        tokens = questions.tokenizer("What is the capital of France?", context, return_tensors="pt")
        cls = tokens["input_ids"][0].tolist().index(questions.tokenizer.cls_token_id)
        start = tokens.char_to_token(0, sequence_index=1)
        end = tokens.char_to_token(4, sequence_index=1)

        self.assertIsNone(questions.answer(context, tokens, cls, cls))
        self.assertEqual(questions.answer(context, tokens, start, end), "Paris")
