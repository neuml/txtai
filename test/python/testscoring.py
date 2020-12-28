"""
Scoring module tests
"""

import os
import tempfile
import unittest

from txtai.scoring import Scoring

class TestScoring(unittest.TestCase):
    """
    Scoring tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize test data.
        """

        cls.data = ["US tops 5 million confirmed virus cases",
                    "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
                    "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
                    "The National Park Service warns against sacrificing slower friends in a bear attack",
                    "Maine man wins $1M from $25 lottery ticket",
                    "wins wins wins",
                    "Make huge profits without work, earn up to $100,000 a day"]

        cls.data = [(uid, x, None) for uid, x in enumerate(cls.data)]

    def testBM25(self):
        """
        Test bm25
        """

        self.method("bm25")
        self.weights("bm25")

    def testSIF(self):
        """
        Test sif
        """

        self.method("sif")
        self.weights("sif")

    def testTFIDF(self):
        """
        Test tfidf
        """

        self.method("tfidf")
        self.weights("tfidf")

    def testUnknown(self):
        """
        Test unknown method
        """

        self.assertIsNone(Scoring.create("unknown"))

    def method(self, method, data=None):
        """
        Runs scoring method
        """

        # Derive input data
        data = data if data else self.data

        model = Scoring.create(method)
        model.index(data)

        keys = [k for k, v in sorted(model.idf.items(), key=lambda x: x[1])]

        # Win should be lowest score for all models
        self.assertEqual(keys[0], "wins")

        # Test save/load
        self.assertIsNotNone(self.save(model))

        return model

    def save(self, model):
        """
        Test scoring index save/load
        """

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "scoring")
        os.makedirs(index, exist_ok=True)

        model.save(index)
        model.load(index)

        return model

    def weights(self, method):
        """
        Test standard and tag weighted scores
        """

        document = (1, ["bear", "wins"], None)

        model = self.method(method)
        weights = model.weights(document)

        # Default weights
        self.assertNotEqual(weights[0], weights[1])

        data = self.data[:]

        uid, text, _ = data[3]
        data[3] = (uid, text, "wins")

        model = self.method(method, data)
        weights = model.weights(document)

        # Modified weights
        self.assertEqual(weights[0], weights[1])
