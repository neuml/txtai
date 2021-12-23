"""
Embeddings API module tests
"""

import os
import tempfile
import unittest
import urllib.parse

from unittest.mock import patch

from fastapi.testclient import TestClient

from txtai.api import API, app, start

# Configuration for a read/write embeddings index
INDEX = """
# Index file path
path: %s

# Allow indexing of documents
writable: True

# Embeddings settings
embeddings:
    path: sentence-transformers/nli-mpnet-base-v2

# Extractor settings
extractor:
    path: distilbert-base-cased-distilled-squad
"""

# Configuration for a read-only embeddings index
READONLY = """
# Index file path
path: %s

# Allow indexing of documents
writable: False
"""


class TestEmbeddings(unittest.TestCase):
    """
    API tests for embeddings indices.
    """

    @staticmethod
    @patch.dict(os.environ, {"CONFIG": os.path.join(tempfile.gettempdir(), "testapi.yml"), "API_CLASS": "txtai.api.API"})
    def start(full):
        """
        Starts a mock FastAPI client.

        Args:
            full: true if full api configuration should be loaded, otherwise a read-only configuration is used
        """

        config = os.path.join(tempfile.gettempdir(), "testapi.yml")
        index = os.path.join(tempfile.gettempdir(), "testapi")

        with open(config, "w", encoding="utf-8") as output:
            output.write((INDEX if full else READONLY) % (index))

        client = TestClient(app)
        start()

        return client

    @classmethod
    def setUpClass(cls):
        """
        Create API client on creation of class.
        """

        cls.client = TestEmbeddings.start(True)

        cls.data = [
            "US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day",
        ]

        # Index data
        cls.client.post("add", json=[{"id": x, "text": row} for x, row in enumerate(cls.data)])
        cls.client.get("index")

    def testCount(self):
        """
        Test count via API
        """

        self.assertEqual(self.client.get("count").json(), 6)

    def testDelete(self):
        """
        Test delete via API
        """

        # Delete best match
        ids = self.client.post("delete", json=[4]).json()
        self.assertEqual(ids, [4])

        # Search for best match
        query = urllib.parse.quote("feel good story")
        uid = self.client.get(f"search?query={query}&limit=1").json()[0]["id"]

        self.assertEqual(self.client.get("count").json(), 5)
        self.assertEqual(uid, 5)

        # Reset data
        self.client.post("add", json=[{"id": x, "text": row} for x, row in enumerate(self.data)])
        self.client.get("index")

    def testEmpty(self):
        """
        Test empty API configuration
        """

        api = API({})

        self.assertIsNone(api.search("test", None))
        self.assertIsNone(api.batchsearch(["test"], None))
        self.assertIsNone(api.delete(["test"]))
        self.assertIsNone(api.count())
        self.assertIsNone(api.similarity("test", ["test"]))
        self.assertIsNone(api.batchsimilarity(["test"], ["test"]))
        self.assertIsNone(api.transform("test"))
        self.assertIsNone(api.batchtransform(["test"]))
        self.assertIsNone(api.extract(["test"], ["test"]))

    def testExtractor(self):
        """
        Test qa extraction via API
        """

        data = [
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

        questions = ["What team won the game?", "What was score?"]

        execute = lambda query: self.client.post(
            "extract",
            json={"queue": [{"name": question, "query": query, "question": question, "snippet": False} for question in questions], "texts": data},
        ).json()

        answers = execute("Red Sox - Blue Jays")
        self.assertEqual("Blue Jays", answers[0]["answer"])
        self.assertEqual("2-1", answers[1]["answer"])

        # Ad-hoc questions
        question = "What hockey team won?"

        answers = self.client.post(
            "extract", json={"queue": [{"name": question, "query": question, "question": question, "snippet": False}], "texts": data}
        ).json()
        self.assertEqual("Flyers", answers[0]["answer"])

    def testSearch(self):
        """
        Test search via API
        """

        query = urllib.parse.quote("feel good story")
        uid = self.client.get(f"search?query={query}&limit=1").json()[0]["id"]
        self.assertEqual(uid, 4)

    def testSearchBatch(self):
        """
        Test batch search via API
        """

        results = self.client.post("batchsearch", json={"queries": ["feel good story", "climate change"], "limit": 1}).json()

        uids = [result[0]["id"] for result in results]
        self.assertEqual(uids, [4, 1])

    def testSimilarity(self):
        """
        Test similarity via API
        """

        uid = self.client.post("similarity", json={"query": "feel good story", "texts": self.data}).json()[0]["id"]

        self.assertEqual(uid, 4)

    def testSimilarityBatch(self):
        """
        Test batch similarity via API
        """

        results = self.client.post("batchsimilarity", json={"queries": ["feel good story", "climate change"], "texts": self.data}).json()

        uids = [result[0]["id"] for result in results]
        self.assertEqual(uids, [4, 1])

    def testTransform(self):
        """
        Test embeddings transform via API
        """

        self.assertEqual(len(self.client.get("transform?text=testembed").json()), 768)

    def testTransformBatch(self):
        """
        Test batch embeddings transform via API
        """

        embeddings = self.client.post("batchtransform", json=self.data).json()

        self.assertEqual(len(embeddings), len(self.data))
        self.assertEqual(len(embeddings[0]), 768)

    def testUpsert(self):
        """
        Test upsert via API
        """

        # Update data
        self.client.post("add", json=[{"id": 0, "text": "Feel good story: baby panda born"}])
        self.client.get("upsert")

        # Search for best match
        query = urllib.parse.quote("feel good story")
        uid = self.client.get(f"search?query={query}&limit=1").json()[0]["id"]

        self.assertEqual(uid, 0)

        # Reset data
        self.client.post("add", json=[{"id": x, "text": row} for x, row in enumerate(self.data)])
        self.client.get("index")

    def testViewOnly(self):
        """
        Test read-only API instance
        """

        # Re-create read-only model
        self.client = TestEmbeddings.start(False)

        # Test search
        query = urllib.parse.quote("feel good story")
        uid = self.client.get(f"search?query={query}&limit=1").json()[0]["id"]
        self.assertEqual(uid, 4)

        # Test similarity
        uid = self.client.post("similarity", json={"query": "feel good story", "texts": self.data}).json()[0]["id"]

        self.assertEqual(uid, 4)
