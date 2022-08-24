"""
Scoring module tests
"""

import os
import tempfile
import unittest

from txtai.scoring import ScoringFactory


class TestScoring(unittest.TestCase):
    """
    Scoring tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize test data.
        """

        cls.data = [
            "US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "wins wins wins",
            "Make huge profits without work, earn up to $100,000 a day",
        ]

        cls.data = [(uid, x, None) for uid, x in enumerate(cls.data)]

    def testBM25(self):
        """
        Test bm25
        """

        self.runTests("bm25")

    def testSIF(self):
        """
        Test sif
        """

        self.runTests("sif")

    def testTFIDF(self):
        """
        Test tfidf
        """

        self.runTests("tfidf")

    def testUnknown(self):
        """
        Test unknown method
        """

        self.assertIsNone(ScoringFactory.create("unknown"))

    def runTests(self, method):
        """
        Runs a series of tests for a scoring method.

        Args:
            method: scoring method
        """

        config = {"method": method}

        self.index(config)
        self.weights(config)
        self.search(config)
        self.content(config)

    def index(self, config, data=None):
        """
        Test scoring index method.

        Args:
            config: scoring config
            data: data to index with scoring method

        Returns:
            Scoring model
        """

        # Derive input data
        data = data if data else self.data

        model = ScoringFactory.create(config)
        model.index(data)

        keys = [k for k, v in sorted(model.idf.items(), key=lambda x: x[1])]

        # Test count
        self.assertEqual(model.count(), len(data))

        # Win should be lowest score for all models
        self.assertEqual(keys[0], "wins")

        # Test save/load
        self.assertIsNotNone(self.save(model))

        # Test search returns none when terms disabled (default)
        self.assertIsNone(model.search("query"))

        return model

    def save(self, model):
        """
        Test scoring index save/load.

        Args:
            model: Scoring model

        Returns:
            Scoring model
        """

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "scoring")
        os.makedirs(index, exist_ok=True)

        model.save(f"{index}/scoring")
        model.load(f"{index}/scoring")

        return model

    def weights(self, config):
        """
        Test standard and tag weighted scores.

        Args:
            config: scoring config
        """

        document = (1, ["bear", "wins"], None)

        model = self.index(config)
        weights = model.weights(document[1])

        # Default weights
        self.assertNotEqual(weights[0], weights[1])

        data = self.data[:]

        uid, text, _ = data[3]
        data[3] = (uid, text, "wins")

        model = self.index(config, data)
        weights = model.weights(document[1])

        # Modified weights
        self.assertEqual(weights[0], weights[1])

    def search(self, config):
        """
        Test scoring search.

        Args:
            method: scoring method
        """

        model = ScoringFactory.create({**config, **{"terms": True}})
        model.index(self.data)

        # Run search and validate correct result returned
        index, _ = model.search("bear", 1)[0]
        self.assertEqual(index, 3)

    def content(self, config):
        """
        Test scoring search with content.

        Args:
            config: scoring config
        """

        model = ScoringFactory.create({**config, **{"terms": True, "content": True}})
        model.index(self.data)

        # Test text with content
        text = "Great news today"
        model.index([(model.total, text, None)])

        # Run search and validate correct result returned
        result = model.search("great news", 1)[0]["text"]
        self.assertEqual(result, text)

        # Test reading text from dictionary
        text = "Feel good story: baby panda born"
        model.index([(model.total, {"text": text}, None)])

        # Run search and validate correct result returned
        result = model.search("feel good story", 1)[0]["text"]
        self.assertEqual(result, text)
