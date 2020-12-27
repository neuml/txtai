"""
API module tests
"""

import os
import tempfile
import unittest

from unittest.mock import patch

import numpy as np

from fastapi.testclient import TestClient

from txtai.api import API, app, start

# Full API configuration with similarity search, extractive QA and zero-shot labeling.
FULL = """
# Index file path
path: %s

# Allow indexing of documents
writable: True

# Embeddings settings
embeddings:
    method: transformers
    path: sentence-transformers/bert-base-nli-mean-tokens

# Extractor settings
extractor:
    path: distilbert-base-cased-distilled-squad

# Labels settings
labels:
    path: squeezebert/squeezebert-mnli
"""

# Configuration that reads an existing similarity search index
READONLY = """
# Index file path
path: %s

# Allow indexing of documents
writable: False
"""

class TestAPI(unittest.TestCase):
    """
    API tests
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

        with open(config, "w") as output:
            output.write((FULL if full else READONLY) % (index))

        client = TestClient(app)
        start()

        return client

    @classmethod
    def setUpClass(cls):
        """
        Create API client on creation of class
        """

        cls.client = TestAPI.start(True)

    def testEmbeddings(self):
        """
        Test embeddings via API
        """

        data = ["US tops 5 million confirmed virus cases",
                "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
                "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
                "The National Park Service warns against sacrificing slower friends in a bear attack",
                "Maine man wins $1M from $25 lottery ticket",
                "Make huge profits without work, earn up to $100,000 a day"]

        # Test similarity
        uid = np.argmax(self.client.post("similarity", json={
            "search": "feel good story",
            "data": data
        }).json())

        self.assertEqual(data[uid], data[4])

        # Test indexing
        self.client.post("add", json=[{"id": x, "text": row} for x, row in enumerate(data)])
        self.client.get("index")

        # Test search
        uid = self.client.get("search?q=feel%20good%20story&n=1").json()[0][0]
        self.assertEqual(data[uid], data[4])

        # Test embeddings
        self.assertIsNotNone(self.client.get("embeddings?t=testembed").json())

    def testEmpty(self):
        """
        Test empty API configuration
        """

        api = API({})

        self.assertIsNone(api.search("test", None))
        self.assertIsNone(api.similarity("test", ["test"]))
        self.assertIsNone(api.transform("test"))
        self.assertIsNone(api.extract(["test"], ["test"]))
        self.assertIsNone(api.label("test", ["test"]))

    def testExtractor(self):
        """
        Test qa extraction via API
        """

        sections = ["Giants hit 3 HRs to down Dodgers",
                    "Giants 5 Dodgers 4 final",
                    "Dodgers drop Game 2 against the Giants, 5-4",
                    "Blue Jays 2 Red Sox 1 final",
                    "Red Sox lost to the Blue Jays, 2-1",
                    "Blue Jays at Red Sox is over. Score: 2-1",
                    "Phillies win over the Braves, 5-0",
                    "Phillies 5 Braves 0 final",
                    "Final: Braves lose to the Phillies in the series opener, 5-0",
                    "Final score: Flyers 4 Lightning 1",
                    "Flyers 4 Lightning 1 final",
                    "Flyers win 4-1"]

        # Add unique id to each section to assist with qa extraction
        sections = [{"id": uid, "text": section} for uid, section in enumerate(sections)]

        questions = ["What team won the game?", "What was score?"]

        execute = lambda query: self.client.post("extract", json={
            "documents": sections,
            "queue": [{"name": question, "query": query, "question": question, "snippet": False} for question in questions]
        }).json()

        answers = execute("Red Sox - Blue Jays")
        self.assertEqual("Blue Jays", answers[0][1])
        self.assertEqual("2-1", answers[1][1])

        # Ad-hoc questions
        question = "What hockey team won?"

        answers = self.client.post("extract", json={
            "documents": sections,
            "queue": [{"name": question, "query": question, "question": question, "snippet": False}]
        }).json()
        self.assertEqual("Flyers", answers[0][1])

    def testLabels(self):
        """
        Test labels via API
        """

        labels = self.client.post("label", json={
            "text": "this is the best sentence ever",
            "labels": ["positive", "negative"]
        }).json()

        self.assertEqual(labels[0][0], "positive")

    def testReadOnly(self):
        """
        Test read-only API instance
        """

        # Re-create read-only model
        self.client = TestAPI.start(False)

        # Test search
        uid = self.client.get("search?q=feel%20good%20story&n=1").json()[0][0]
        self.assertEqual(uid, 4)
