"""
Hybrid module tests
"""

import random
import unittest

from txtai.embeddings.search.hybrid import LogOdds


def calibrate_reference(raw):
    """
    Reference implementation of LogOdds.calibrate with the mean recomputed
    for every element (the pre-optimization behavior).

    Args:
        raw: list of raw dense cosine scores

    Returns:
        (median, alpha) calibration parameters
    """

    median, alpha = 0.0, 1.0

    array = [s for s in raw if s > 0]
    if array:
        median = sorted(array)[len(array) // 2]
        std = (
            sum((x - sum(array) / len(array)) ** 2 for x in array) / len(array)
        ) ** 0.5
        alpha = 1.0 / std if std > 0 else 1.0

    return median, alpha


class TestHybrid(unittest.TestCase):
    """
    Hybrid scoring tests.
    """

    def testCalibrateEmpty(self):
        """
        Test calibration with empty input
        """

        self.assertEqual(LogOdds().calibrate([]), (0.0, 1.0))

    def testCalibrateNonPositive(self):
        """
        Test calibration with no positive scores
        """

        self.assertEqual(LogOdds().calibrate([-1.0, 0.0, -0.5]), (0.0, 1.0))

    def testCalibrateSingleton(self):
        """
        Test calibration with a single positive score (zero variance fallback)
        """

        self.assertEqual(LogOdds().calibrate([0.7]), (0.7, 1.0))

    def testCalibrateConstant(self):
        """
        Test calibration with constant positive scores (zero variance fallback)
        """

        self.assertEqual(LogOdds().calibrate([0.5, 0.5, 0.5]), (0.5, 1.0))

    def testCalibrateMatchesReference(self):
        """
        Test calibration matches the reference (pre-optimization) formula exactly
        """

        random.seed(42)

        for _ in range(50):
            raw = [random.uniform(-1.0, 1.0) for _ in range(random.randint(0, 100))]

            self.assertEqual(LogOdds().calibrate(raw), calibrate_reference(raw))

    def testFusionMatchesReference(self):
        """
        Test full log-odds fusion matches the reference formula exactly
        """

        random.seed(1337)

        scorer = LogOdds()

        for _ in range(25):
            n = random.randint(1, 50)
            dense = [(i, random.uniform(-1.0, 1.0)) for i in range(n)]
            sparse = [(i, random.uniform(0.0, 1.0)) for i in range(n)]

            uids, denseraw = scorer.rawscores((dense, sparse), [0.5, 0.5])

            median, alpha = scorer.calibrate(denseraw)
            expected = scorer.fuse(uids, [0.5, 0.5], *calibrate_reference(denseraw))

            self.assertEqual(median, calibrate_reference(denseraw)[0])
            self.assertEqual(alpha, calibrate_reference(denseraw)[1])
            self.assertEqual(scorer.fuse(uids, [0.5, 0.5], median, alpha), expected)

    def testFusionOrder(self):
        """
        Test fused results are sorted by score descending and capped at limit
        """

        dense = [(i, 0.1 * (i + 1)) for i in range(10)]
        sparse = [(i, 0.9) for i in range(10)]

        result = LogOdds()((dense, sparse), [0.5, 0.5], 5)

        self.assertEqual(len(result), 5)
        self.assertEqual([uid for uid, _ in result], [9, 8, 7, 6, 5])
