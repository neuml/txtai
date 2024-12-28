"""
Scoring module tests
"""

import os
import tempfile
import unittest

from unittest.mock import patch

from txtai.scoring import ScoringFactory, Scoring


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

    def testCustom(self):
        """
        Test custom method
        """

        self.runTests("txtai.scoring.BM25")

    def testCustomNotFound(self):
        """
        Test unresolvable custom method
        """

        with self.assertRaises(ImportError):
            ScoringFactory.create("notfound.scoring")

    def testNotImplemented(self):
        """
        Test exceptions for non-implemented methods
        """

        scoring = Scoring()

        self.assertRaises(NotImplementedError, scoring.insert, None, None)
        self.assertRaises(NotImplementedError, scoring.delete, None)
        self.assertRaises(NotImplementedError, scoring.weights, None)
        self.assertRaises(NotImplementedError, scoring.search, None, None)
        self.assertRaises(NotImplementedError, scoring.batchsearch, None, None, None)
        self.assertRaises(NotImplementedError, scoring.count)
        self.assertRaises(NotImplementedError, scoring.load, None)
        self.assertRaises(NotImplementedError, scoring.save, None)
        self.assertRaises(NotImplementedError, scoring.close)
        self.assertRaises(NotImplementedError, scoring.hasterms)
        self.assertRaises(NotImplementedError, scoring.isnormalized)

    @patch("sqlalchemy.orm.Query.params")
    def testPGText(self, query):
        """
        Test PGText
        """

        # Mock database query
        query.return_value = [(3, 1.0)]

        # Create scoring
        path = os.path.join(tempfile.gettempdir(), "pgtext.sqlite")
        scoring = ScoringFactory.create({"method": "pgtext", "url": f"sqlite:///{path}", "schema": "txtai"})
        scoring.index((uid, {"text": text}, tags) for uid, text, tags in self.data)

        # Run search and validate correct result returned
        index, _ = scoring.search("bear", 1)[0]
        self.assertEqual(index, 3)

        # Run batch search
        index, _ = scoring.batchsearch(["bear"], 1)[0][0]
        self.assertEqual(index, 3)

        # Validate save/load/delete
        scoring.save(None)
        scoring.load(None)

        # Validate count
        self.assertEqual(scoring.count(), len(self.data))

        # Test delete
        scoring.delete([0])
        self.assertEqual(scoring.count(), len(self.data) - 1)

        # PGText is a normalized terms index
        self.assertTrue(scoring.hasterms() and scoring.isnormalized())
        self.assertIsNone(scoring.weights("This is a test".split()))

        # Close scoring
        scoring.close()

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

    def runTests(self, method):
        """
        Runs a series of tests for a scoring method.

        Args:
            method: scoring method
        """

        config = {"method": method}

        self.index(config)
        self.upsert(config)
        self.weights(config)
        self.search(config)
        self.delete(config)
        self.normalize(config)
        self.content(config)
        self.empty(config)
        self.copy(config)
        self.settings(config)

    def index(self, config, data=None):
        """
        Test scoring index method.

        Args:
            config: scoring config
            data: data to index with scoring method

        Returns:
            scoring
        """

        # Derive input data
        data = data if data else self.data

        scoring = ScoringFactory.create(config)
        scoring.index(data)

        keys = [k for k, v in sorted(scoring.idf.items(), key=lambda x: x[1])]

        # Test count
        self.assertEqual(scoring.count(), len(data))

        # Win should be lowest score
        self.assertEqual(keys[0], "wins")

        # Test save/load
        self.assertIsNotNone(self.save(scoring, config, f"scoring.{config['method']}.index"))

        # Test search returns none when terms disabled (default)
        self.assertIsNone(scoring.search("query"))

        return scoring

    def upsert(self, config):
        """
        Test scoring upsert method
        """

        scoring = ScoringFactory.create({**config, **{"tokenizer": {"alphanum": True, "stopwords": True}}})
        scoring.upsert(self.data)

        # Test count
        self.assertEqual(scoring.count(), len(self.data))

        # Test stop word is removed
        self.assertFalse("and" in scoring.idf)

    def save(self, scoring, config, name):
        """
        Test scoring index save/load.

        Args:
            scoring: scoring index
            config: scoring config
            name: output file name

        Returns:
            scoring
        """

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "scoring")
        os.makedirs(index, exist_ok=True)

        # Save scoring instance
        scoring.save(f"{index}/{name}")

        # Reload scoring instance
        scoring = ScoringFactory.create(config)
        scoring.load(f"{index}/{name}")

        return scoring

    def weights(self, config):
        """
        Test standard and tag weighted scores.

        Args:
            config: scoring config
        """

        document = (1, ["bear", "wins"], None)

        scoring = self.index(config)
        weights = scoring.weights(document[1])

        # Default weights
        self.assertNotEqual(weights[0], weights[1])

        data = self.data[:]

        uid, text, _ = data[3]
        data[3] = (uid, text, "wins")

        scoring = self.index(config, data)
        weights = scoring.weights(document[1])

        # Modified weights
        self.assertEqual(weights[0], weights[1])

    def search(self, config):
        """
        Test scoring search.

        Args:
            config: scoring config
        """

        # Create combined config
        config = {**config, **{"terms": True}}

        # Create scoring instance
        scoring = ScoringFactory.create(config)
        scoring.index(self.data)

        # Run search and validate correct result returned
        index, _ = scoring.search("bear", 1)[0]
        self.assertEqual(index, 3)

        # Run batch search
        index, _ = scoring.batchsearch(["bear"], 1)[0][0]
        self.assertEqual(index, 3)

        # Test save/reload
        self.save(scoring, config, f"scoring.{config['method']}.search")

        # Re-run search and validate correct result returned
        index, _ = scoring.search("bear", 1)[0]
        self.assertEqual(index, 3)

    def delete(self, config):
        """
        Test delete.
        """

        # Create combined config
        config = {**config, **{"terms": True, "content": True}}

        # Create scoring instance
        scoring = ScoringFactory.create(config)
        scoring.index(self.data)

        # Run search and validate correct result returned
        index = scoring.search("bear", 1)[0]["id"]

        # Delete result and validate the query no longer returns results
        scoring.delete([index])
        self.assertFalse(scoring.search("bear", 1))

        # Save and validate count
        self.save(scoring, config, f"scoring.{config['method']}.delete")
        self.assertEqual(scoring.count(), len(self.data) - 1)

    def normalize(self, config):
        """
        Test scoring search with normalized scores.

        Args:
            method: scoring method
        """

        scoring = ScoringFactory.create({**config, **{"terms": True, "normalize": True}})
        scoring.index(self.data)

        # Run search and validate correct result returned
        index, score = scoring.search(self.data[3][1], 1)[0]
        self.assertEqual(index, 3)
        self.assertEqual(score, 1.0)

    def content(self, config):
        """
        Test scoring search with content.

        Args:
            config: scoring config
        """

        scoring = ScoringFactory.create({**config, **{"terms": True, "content": True}})
        scoring.index(self.data)

        # Test text with content
        text = "Great news today"
        scoring.index([(scoring.total, text, None)])

        # Run search and validate correct result returned
        result = scoring.search("great news", 1)[0]["text"]
        self.assertEqual(result, text)

        # Test reading text from dictionary
        text = "Feel good story: baby panda born"
        scoring.index([(scoring.total, {"text": text}, None)])

        # Run search and validate correct result returned
        result = scoring.search("feel good story", 1)[0]["text"]
        self.assertEqual(result, text)

    def empty(self, config):
        """
        Test scoring index properly handles an index call when no data present.

        Args:
            config: scoring config
        """

        # Create scoring index with no data
        scoring = ScoringFactory.create(config)
        scoring.index([])

        # Assert index call returns and index has a count of 0
        self.assertEqual(scoring.total, 0)

    def copy(self, config):
        """
        Test scoring index copy method.
        """

        # Create scoring instance
        scoring = ScoringFactory.create({**config, **{"terms": True}})
        scoring.index(self.data)

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "scoring")
        os.makedirs(index, exist_ok=True)

        # Create file to test replacing existing file
        path = f"{index}/scoring.{config['method']}.copy"
        with open(f"{index}.terms", "w", encoding="utf-8") as f:
            f.write("TEST")

        # Save scoring instance
        scoring.save(path)
        self.assertTrue(os.path.exists(path))

    @patch("sys.byteorder", "big")
    def settings(self, config):
        """
        Test various settings.

        Args:
            config: scoring config
        """

        # Create combined config
        config = {**config, **{"terms": {"cachelimit": 0, "cutoff": 0.25, "wal": True}}}

        # Create scoring instance
        scoring = ScoringFactory.create(config)
        scoring.index(self.data)

        # Save/load index
        self.save(scoring, config, f"scoring.{config['method']}.settings")

        index, _ = scoring.search("bear bear bear wins", 1)[0]
        self.assertEqual(index, 3)

        # Save to same path
        self.save(scoring, config, f"scoring.{config['method']}.settings")

        # Save to different path
        self.save(scoring, config, f"scoring.{config['method']}.move")

        # Validate counts
        self.assertEqual(scoring.count(), len(self.data))
