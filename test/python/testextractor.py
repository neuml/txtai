"""
Extractor module tests
"""

# import platform
import unittest

from txtai.embeddings import Embeddings
from txtai.pipeline import Extractor, Similarity


class TestExtractor(unittest.TestCase):
    """
    Extractor tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Create single extractor instance.
        """

        cls.data = [
            "Giants hit 3 HRs to down Dodgers",
            "Giants 5 Dodgers 4 final",
            "Dodgers drop Game 2 against the Giants, 5-4",
            "Blue Jays beat Red Sox final score 2-1",
            "Red Sox lost to the Blue Jays, 2-1",
            "Blue Jays at Red Sox is over. Score: 2-1",
            "Phillies win over the Braves, 5-0",
            "Phillies 5 Braves 0 final",
            "Final: Braves lose to the Phillies in the series opener, 5-0",
            "Lightning goaltender pulled, lose to Flyers 4-1",
            "Flyers 4 Lightning 1 final",
            "Flyers win 4-1",
        ]

        # Create embeddings model, backed by sentence-transformers & transformers
        cls.embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})

        # Create extractor instance
        cls.extractor = Extractor(cls.embeddings, "distilbert-base-cased-distilled-squad")

    def testAnswer(self):
        """
        Test qa extraction with an answer
        """

        questions = ["What team won the game?", "What was score?"]

        execute = lambda query: self.extractor([(question, query, question, False) for question in questions], self.data)

        answers = execute("Red Sox - Blue Jays")
        self.assertEqual("Blue Jays", answers[0][1])
        self.assertEqual("2-1", answers[1][1])

        # Ad-hoc questions
        question = "What hockey team won?"

        answers = self.extractor([(question, question, question, False)], self.data)
        self.assertEqual("Flyers", answers[0][1])

    def testEmptyQuery(self):
        """
        Tests an empty extractor queries list.
        """

        self.assertEqual(self.extractor.query(None, None), [])

    def testNoAnswer(self):
        """
        Test qa extraction with no answer
        """

        question = ""

        answers = self.extractor([(question, question, question, False)], self.data)
        self.assertIsNone(answers[0][1])

        question = "abcdef"
        answers = self.extractor([(question, question, question, False)], self.data)
        self.assertIsNone(answers[0][1])

    # @unittest.skipIf(platform.system() == "Darwin", "Quantized models not supported on macOS")
    def testQuantize(self):
        """
        Test qa extraction backed by a quantized model
        """

        extractor = Extractor(self.embeddings, "distilbert-base-cased-distilled-squad", True)

        question = "How many home runs?"

        answers = extractor([(question, question, question, True)], self.data)
        self.assertTrue(answers[0][1].startswith("Giants hit 3 HRs"))

    def testSimilarity(self):
        """
        Test qa extraction using a Similarity pipeline to build context
        """

        # Create extractor instance
        extractor = Extractor(Similarity("prajjwal1/bert-medium-mnli"), "distilbert-base-cased-distilled-squad")

        question = "How many home runs?"

        answers = extractor([(question, "HRs", question, True)], self.data)
        self.assertTrue(answers[0][1].startswith("Giants hit 3 HRs"))

    def testSnippet(self):
        """
        Test qa extraction with a full answer snippet
        """

        question = "How many home runs?"

        answers = self.extractor([(question, question, question, True)], self.data)
        self.assertTrue(answers[0][1].startswith("Giants hit 3 HRs"))

    def testSnippetEmpty(self):
        """
        Test snippet method can handle empty parameters
        """

        self.assertEqual(self.extractor.snippet(None, None), None)
