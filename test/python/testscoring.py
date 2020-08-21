"""
Scoring module tests
"""

import os
import tempfile
import unittest

from txtai.scoring import Scoring

class TestScoring(unittest.TestCase):
    """
    Scoring tests.
    """

    def setUp(self):
        """
        Initialize test data.
        """

        self.data = ["US tops 5 million confirmed virus cases",
                     "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
                     "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
                     "The National Park Service warns against sacrificing slower friends in a bear attack",
                     "Maine man wins $1M from $25 lottery ticket",
                     "wins wins wins",
                     "Make huge profits without work, earn up to $100,000 a day"]

        self.data = [(uid, x, None) for uid, x in enumerate(self.data)]

    def testBM25(self):
        """
        Test BM25
        """

        self.method("bm25")

    def testSIF(self):
        """
        Test SIF
        """

        self.method("sif")

    def testTFIDF(self):
        """
        Test tfidf
        """

        self.method("tfidf")

    def testSave(self):
        """
        Test scoring index save/load
        """

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "bm25")
        os.makedirs(index, exist_ok=True)

        model = self.method("bm25")
        model.save(index)
        model.load(index)

    def method(self, method):
        """
        Runs scoring method
        """

        model = Scoring.create(method)
        model.index(self.data)

        keys = [k for k, v in sorted(model.idf.items(), key=lambda x: x[1])]

        # Win should be lowest score for all models
        self.assertEqual(keys[0], "wins")

        return model
