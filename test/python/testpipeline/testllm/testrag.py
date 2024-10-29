"""
RAG module tests
"""

import platform
import unittest

from txtai.embeddings import Embeddings
from txtai.pipeline import Questions, RAG, Similarity


class TestRAG(unittest.TestCase):
    """
    RAG tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create single rag instance.
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

        # Create rag instance
        cls.rag = RAG(cls.embeddings, "distilbert-base-cased-distilled-squad")

    @classmethod
    def tearDownClass(cls):
        """
        Cleanup data.
        """

        if cls.embeddings:
            cls.embeddings.close()

    def testAnswer(self):
        """
        Test qa extraction with an answer
        """

        questions = ["What team won the game?", "What was score?"]

        # pylint: disable=C3001
        execute = lambda query: self.rag([(question, query, question, False) for question in questions], self.data)

        answers = execute("Red Sox - Blue Jays")
        self.assertEqual("Blue Jays", answers[0][1])
        self.assertEqual("2-1", answers[1][1])

        # Ad-hoc questions
        question = "What hockey team won?"

        answers = self.rag([(question, question, question, False)], self.data)
        self.assertEqual("Flyers", answers[0][1])

    def testEmptyQuery(self):
        """
        Test an empty queries list
        """

        self.assertEqual(self.rag.query(None, None), [])

    def testNoAnswer(self):
        """
        Test qa extraction with no answer
        """

        question = ""

        answers = self.rag([(question, question, question, False)], self.data)
        self.assertIsNone(answers[0][1])

        question = "abcdef"
        answers = self.rag([(question, question, question, False)], self.data)
        self.assertIsNone(answers[0][1])

    @unittest.skipIf(platform.system() == "Darwin", "Quantized models not supported on macOS")
    def testQuantize(self):
        """
        Test qa extraction backed by a quantized model
        """

        rag = RAG(self.embeddings, "distilbert-base-cased-distilled-squad", True)

        question = "How many home runs?"

        answers = rag([(question, question, question, True)], self.data)
        self.assertTrue(answers[0][1].startswith("Giants hit 3 HRs"))

    def testOutputs(self):
        """
        Test output formatting rules
        """

        question = "How many home runs?"

        # Test flatten to list of answers
        rag = RAG(self.embeddings, "distilbert-base-cased-distilled-squad", output="flatten")
        answers = rag([(question, question, question, True)], self.data)
        self.assertTrue(answers[0].startswith("Giants hit 3 HRs"))

        # Test reference field
        rag = RAG(self.embeddings, "distilbert-base-cased-distilled-squad", output="reference")
        answers = rag([(question, question, question, True)], self.data)
        self.assertTrue(self.data[answers[0][2]].startswith("Giants hit 3 HRs"))

    def testPrompt(self):
        """
        Test a user prompt with templating
        """

        embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": True})
        embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        rag = RAG(
            embeddings,
            "google/flan-t5-small",
            template="""
              Answer the following question and return a number.
              Question: {question}
              Context:{context}""",
            output="flatten",
        )

        self.assertEqual(rag("How many HRs"), "3")

    def testPromptTemplates(self):
        """
        Test system and user prompt templates
        """

        rag = RAG(
            self.embeddings,
            "sshleifer/tiny-gpt2",
            system="You are a friendly assistant",
            template="""
              Answer the following question and return a number.
              Question: {question}
              Context:{context}""",
        )

        prompts = rag.prompts(["How many HRs?"], [self.data])[0]
        self.assertEqual([x["role"] for x in prompts], ["system", "user"])

    def testSearch(self):
        """
        Test qa extraction with an embeddings search for context
        """

        embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": True})
        embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        rag = RAG(embeddings, "distilbert-base-cased-distilled-squad")

        question = "How many home runs?"

        answers = rag([(question, question, question, True)])
        self.assertTrue(answers[0][1].startswith("Giants hit 3 HRs"))

    def testSimilarity(self):
        """
        Test qa extraction using a Similarity pipeline to build context
        """

        # Create rag instance
        rag = RAG(Similarity("prajjwal1/bert-medium-mnli"), Questions("distilbert-base-cased-distilled-squad"))

        question = "How many home runs?"

        answers = rag([(question, "HRs", question, True)], self.data)
        self.assertTrue(answers[0][1].startswith("Giants hit 3 HRs"))

    def testSnippet(self):
        """
        Test qa extraction with a full answer snippet
        """

        question = "How many home runs?"

        answers = self.rag([(question, question, question, True)], self.data)
        self.assertTrue(answers[0][1].startswith("Giants hit 3 HRs"))

    def testSnippetEmpty(self):
        """
        Test snippet method can handle empty parameters
        """

        self.assertEqual(self.rag.snippets(["name"], [None], [None], [None]), [("name", None)])

    def testStringInput(self):
        """
        Test with single string input
        """

        result = self.rag("How many home runs?", self.data)
        self.assertEqual(result["answer"], "3")

    def testTasks(self):
        """
        Test loading models with task parameter
        """

        for task, model in [
            ("language-generation", "hf-internal-testing/tiny-random-gpt2"),
            ("sequence-sequence", "hf-internal-testing/tiny-random-t5"),
        ]:
            rag = RAG(self.embeddings, model, task=task)
            self.assertIsNotNone(rag)
