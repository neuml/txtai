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
        Test a SQuAD2 no-answer prediction alongside an answerable question
        """

        questions = Questions("deepset/minilm-uncased-squad2", gpu=False)
        context = "Paris is the capital of France."
        self.assertEqual(
            questions(["Who is the president of Mars?", "What is the capital of France?"], [context, context]),
            [None, "Paris"],
        )
