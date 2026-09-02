"""
Similarity module tests
"""

import unittest

from txtai.pipeline import Similarity


class TestSimilarity(unittest.TestCase):
    """
    Similarity tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create single labels instance.
        """

        cls.data = [
            "US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day",
        ]

        cls.similarity = Similarity("prajjwal1/bert-medium-mnli")

    def testCrossEncoder(self):
        """
        Test cross-encoder similarity model
        """

        similarity = Similarity("cross-encoder/ms-marco-MiniLM-L-2-v2", crossencode=True)
        uid = similarity("Who won the lottery?", self.data)[0][0]
        self.assertEqual(self.data[uid], self.data[4])

    def testCrossEncoderBatch(self):
        """
        Test cross-encoder similarity model with multiple inputs
        """

        similarity = Similarity("cross-encoder/ms-marco-MiniLM-L-2-v2", crossencode=True)
        results = [r[0][0] for r in similarity(["Who won the lottery?", "Where did an iceberg collapse?"], self.data)]
        self.assertEqual(results, [4, 1])

    def testCrossEncoderLabels(self):
        """
        Test cross-encoder ranking with a multi-class model, restricted to a single label via labels
        """

        similarity = Similarity("prajjwal1/bert-medium-mnli", crossencode=True)

        premise = "A dog is running in the park."
        entailment = "An animal is outside."
        contradiction = "The park is empty and there are no animals."

        # Without labels, each candidate is scored by whichever label it individually scored highest
        # on - here the contradiction's own top score outscores the entailment's own top score, so
        # the less relevant candidate wins
        uid = similarity(premise, [entailment, contradiction])[0][0]
        self.assertEqual(uid, 1)

        # Restricting to a single label (LABEL_1 = entailment for this model) compares every
        # candidate on that same label instead, fixing the ranking
        uid = similarity(premise, [entailment, contradiction], labels=["1"])[0][0]
        self.assertEqual(uid, 0)

    def testCrossEncoderLabelsOrder(self):
        """
        Test cross-encoder labels resolve to label ids when the model config's label order differs from id order
        """

        similarity = Similarity("prajjwal1/bert-medium-mnli", crossencode=True)

        premise = "A dog is running in the park."
        texts = ["An animal is outside.", "The park is empty and there are no animals."]

        # Scores restricted to LABEL_1 with the model config as loaded
        expected = similarity(premise, texts, labels=["1"])

        # Rotate the config label mappings so insertion order no longer matches label ids
        config = similarity.pipeline.model.config
        ids = list(config.id2label)
        config.id2label = {x: config.id2label[x] for x in ids[1:] + ids[:1]}
        config.label2id = {label: x for x, label in config.id2label.items()}

        # Verify the same label is scored when requested by name and by id
        self.assertEqual(similarity(premise, texts, labels=["LABEL_1"]), expected)
        self.assertEqual(similarity(premise, texts, labels=["1"]), expected)

    def testCrossEncoderLabelsInvalid(self):
        """
        Test cross-encoder raises an error when no requested label matches the model
        """

        similarity = Similarity("prajjwal1/bert-medium-mnli", crossencode=True)
        with self.assertRaises(ValueError):
            similarity("A dog is running in the park.", ["An animal is outside."], labels=["not-a-real-label"])

    def testLateEncoder(self):
        """
        Test late-encoder similarity model
        """

        similarity = Similarity("neuml/pylate-bert-tiny", lateencode=True)
        uid = similarity("Who won the lottery?", self.data)[0][0]
        self.assertEqual(self.data[uid], self.data[4])

        # Test encode method
        # pylint: disable=E1101
        self.assertEqual(similarity.encode(["Who won the lottery?"], "data").shape, (1, 8, 128))

    def testLateEncoderBatch(self):
        """
        Test late-encoder similarity model with multiple inputs
        """

        similarity = Similarity("neuml/colbert-bert-tiny", lateencode=True)
        queries = ["Who won the lottery?", "Where did an iceberg collapse?"]
        results = similarity(queries, self.data)
        self.assertEqual([r[0][0] for r in results], [4, 1])

        # Test scores don't change with query batch padding
        for query, scores in zip(queries, results):
            alone = dict(similarity(query, self.data))
            for uid, score in scores:
                self.assertAlmostEqual(score, alone[uid], places=4)

    def testLateEncoderPadding(self):
        """
        Test late-encoder scores don't change with data batch padding
        """

        texts = self.data + ["Short text."]
        queries = ["Who won the lottery?", "Where did an iceberg collapse?"]

        # Center token vectors on a fixed collection mean, this is batch independent and makes negative token similarities common
        similarity = Similarity("neuml/colbert-bert-tiny", lateencode=True)
        vectors = similarity.encode(texts, "data")
        mean = vectors[vectors.abs().sum(dim=-1) > 0].mean(dim=0).cpu().numpy()
        similarity = Similarity("neuml/colbert-bert-tiny", lateencode=True, vectors={"center": {"scope": "collection", "mean": mean}})

        # Each text encoded alone has no padding, scores must match the exact MaxSim over true tokens
        for query in queries:
            scores, vectors = dict(similarity(query, texts)), similarity.encode([query], "query")[0]
            for uid, text in enumerate(texts):
                exact = (vectors @ similarity.encode([text], "data")[0].T).max(dim=1).values.mean().item()
                self.assertAlmostEqual(scores[uid], exact, places=4)

    def testSimilarity(self):
        """
        Test similarity with single query
        """

        uid = self.similarity("feel good story", self.data)[0][0]
        self.assertEqual(self.data[uid], self.data[4])

    def testSimilarityBatch(self):
        """
        Test similarity with multiple queries
        """

        results = [r[0][0] for r in self.similarity(["feel good story", "climate change"], self.data)]
        self.assertEqual(results, [4, 1])

    def testSimilarityFixed(self):
        """
        Test similarity with a fixed label text classification model
        """

        similarity = Similarity(dynamic=False)

        # Test with query as label text and label id
        self.assertLessEqual(similarity("negative", ["This is the best sentence ever"])[0][1], 0.1)
        self.assertLessEqual(similarity("0", ["This is the best sentence ever"])[0][1], 0.1)

    def testSimilarityLong(self):
        """
        Test similarity with long text
        """

        uid = self.similarity("other", ["Very long text " * 1000, "other text"])[0][0]
        self.assertEqual(uid, 1)
