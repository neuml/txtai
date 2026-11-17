"""
Hybrid score calibration tests
"""

import math
import unittest

from txtai.embeddings.search.hybrid import Hybrid, LogOdds
from txtai.scoring import ScoringFactory


class TestHybrid(unittest.TestCase):
    """
    Deterministic hybrid fusion tests without model inference.
    """

    def testEvenMedian(self):
        """
        Even candidate sets use the midpoint of the two central scores.
        """

        for scores, expected in [([0.8, 0.2], 0.5), ([0.9, 0.1, 0.3, 0.5], 0.4), ([0.0, -1.0, 0.8, 0.2], 0.5)]:
            with self.subTest(scores=scores):
                median, _ = LogOdds().calibrate(scores)
                self.assertAlmostEqual(median, expected)

    def testCalibrationControls(self):
        """
        Odd, tied, singleton and absent positive scores preserve calibration.
        """

        for scores, expected in [([0.9, 0.1, 0.5], 0.5), ([0.4, 0.4], 0.4), ([0.3], 0.3), ([], 0.0), ([0.0, -1.0], 0.0)]:
            with self.subTest(scores=scores):
                median, alpha = LogOdds().calibrate(scores)
                self.assertAlmostEqual(median, expected)
                self.assertGreater(alpha, 0.0)

    def testEvenMedianFusion(self):
        """
        Median calibration preserves dense evidence against sparse-only matches.
        """

        scoring = ScoringFactory.create({"method": "bm25", "normalize": "bb25"})
        fusion = Hybrid(scoring)
        results = fusion(([("a", 0.8), ("b", 0.2)], [("c", 0.55), ("a", 0.5)]), [0.5, 0.5], 3)

        self.assertEqual([uid for uid, _ in results], ["a", "c", "b"])
        scores = dict(results)
        # Dense a has standardized logit +1; its neutral sparse probability contributes zero.
        self.assertAlmostEqual(scores["a"], 1 / (1 + math.exp(-math.sqrt(2) / 2)))
        self.assertAlmostEqual(scores["b"], 1 / (1 + math.exp(0.5)))
        self.assertAlmostEqual(scores["c"], 1 / (1 + math.exp(-0.5 * math.log(0.55 / 0.45))))
