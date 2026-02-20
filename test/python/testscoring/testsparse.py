"""
Sparse module tests
"""

import os
import platform
import tempfile
import unittest

from unittest.mock import patch

from txtai.scoring import ScoringFactory


# pylint: disable=R0904
class TestSparse(unittest.TestCase):
    """
    Sparse vector scoring tests.
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
            "Make huge profits without work, earn up to $100,000 a day",
        ]

        cls.data = [(uid, x, None) for uid, x in enumerate(cls.data)]

    def testGeneral(self):
        """
        Test general sparse vector operations
        """

        # Models cache
        models = {}

        # Test sparse scoring
        scoring = ScoringFactory.create({"method": "sparse", "path": "sparse-encoder-testing/splade-bert-tiny-nq"}, models=models)
        scoring.index((uid, {"text": text}, tags) for uid, text, tags in self.data)

        # Run search and validate correct result returned
        index, _ = scoring.search("lottery ticket", 1)[0]
        self.assertEqual(index, 4)

        # Run batch search
        index, _ = scoring.batchsearch(["lottery ticket"], 1)[0][0]
        self.assertEqual(index, 4)

        # Validate count
        self.assertEqual(scoring.count(), len(self.data))

        # Test delete
        scoring.delete([4])
        self.assertEqual(scoring.count(), len(self.data) - 1)

        # Run search after delete
        index, _ = scoring.search("lottery ticket", 1)[0]
        self.assertEqual(index, 5)

        # Sparse vectors is a normalized sparse index
        self.assertTrue(scoring.issparse() and scoring.isnormalized() and not scoring.isbayes())
        self.assertIsNone(scoring.weights("This is a test".split()))

        # Close scoring
        scoring.close()

        # Test model caching
        scoring = ScoringFactory.create({"method": "sparse", "path": "sparse-encoder-testing/splade-bert-tiny-nq"}, models=models)
        self.assertIsNotNone(scoring.model)
        scoring.close()

    def testEmpty(self):
        """
        Test empty sparse vectors
        """

        scoring = ScoringFactory.create({"method": "sparse", "path": "sparse-encoder-testing/splade-bert-tiny-nq"})
        scoring.upsert((uid, {"text": text}, tags) for uid, text, tags in self.data)
        self.assertEqual(scoring.count(), len(self.data))

    @unittest.skipIf(platform.system() == "Darwin", "Torch memory sharing not supported on macOS")
    @patch("torch.cuda.device_count")
    def testGPU(self, count):
        """
        Test sparse vectors with GPU encoding
        """

        # Mock accelerator count
        count.return_value = 2

        # Test multiple gpus
        scoring = ScoringFactory.create({"method": "sparse", "path": "sparse-encoder-testing/splade-bert-tiny-nq", "gpu": "all"})
        self.assertIsNotNone(scoring)
        scoring.close()

    def testBayes(self):
        """
        Test BB25 Bayesian normalization for sparse scoring
        """

        config = {
            "method": "sparse",
            "path": "sparse-encoder-testing/splade-bert-tiny-nq",
            "normalize": "bb25",
        }
        scoring = ScoringFactory.create(config)
        scoring.index((uid, {"text": text}, tags) for uid, text, tags in self.data)

        # Verify Bayesian mode flags
        self.assertTrue(scoring.isbayes())
        self.assertTrue(scoring.isnormalized())

        # Search and validate scores are calibrated probabilities in [0, 1]
        results = scoring.search("lottery ticket", 3)
        self.assertGreater(len(results), 0)
        for _, score in results:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        # Batch search
        results = scoring.batchsearch(["lottery ticket", "ice shelf"], 3)
        self.assertEqual(len(results), 2)
        for query_results in results:
            for _, score in query_results:
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

        scoring.close()

    def testBayesDict(self):
        """
        Test BB25 normalization with dict config
        """

        config = {
            "method": "sparse",
            "path": "sparse-encoder-testing/splade-bert-tiny-nq",
            "normalize": {"method": "bb25", "alpha": 2.0},
        }
        scoring = ScoringFactory.create(config)
        scoring.index((uid, {"text": text}, tags) for uid, text, tags in self.data)

        self.assertTrue(scoring.isbayes())

        results = scoring.search("lottery ticket", 3)
        self.assertGreater(len(results), 0)
        for _, score in results:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        scoring.close()

    def testBayesNonBayes(self):
        """
        Test that non-Bayesian string normalize values do not activate Bayesian mode
        """

        config = {
            "method": "sparse",
            "path": "sparse-encoder-testing/splade-bert-tiny-nq",
            "normalize": "default",
        }
        scoring = ScoringFactory.create(config)
        self.assertFalse(scoring.isbayes())
        scoring.close()

    def testIVFFlat(self):
        """
        Test sparse vectors with IVFFlat clustering
        """

        # Expand dataset
        data = self.data * 1000

        # Test higher volume IVFFlat index with clustering
        config = {
            "method": "sparse",
            "vectormethod": "sentence-transformers",
            "path": "sparse-encoder-testing/splade-bert-tiny-nq",
            "ivfsparse": {"sample": 1.0},
        }
        scoring = ScoringFactory.create(config)
        scoring.index((uid, {"text": text}, tags) for uid, text, tags in data)

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "scoring")
        os.makedirs(index, exist_ok=True)

        # Save scoring instance
        scoring.save(f"{index}/scoring.sparse.index")

        # Reload scoring instance
        scoring = ScoringFactory.create(config)
        scoring.load(f"{index}/scoring.sparse.index")

        # Run search and validate correct result returned
        results = scoring.search("lottery ticket", 1)
        self.assertGreater(len(results), 0)
        scoring.close()
