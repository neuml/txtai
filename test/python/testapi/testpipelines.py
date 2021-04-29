"""
Pipeline API module tests
"""

import os
import tempfile
import unittest
import urllib

from unittest.mock import patch

from fastapi.testclient import TestClient

from txtai.api import API, app, start

# pylint: disable = C0411
from utils import Utils

# Configuration for pipelines
PIPELINES = """
# Labels settings
labels:
    path: prajjwal1/bert-medium-mnli

# Text segmentation
segmentation:
    sentences: true

# Enable pipeline similarity backed by zero shot classifier
similarity:

# Summarization
summary:
    path: t5-small

# Text extraction
textractor:

# Transcription
transcription:

# Translation:
translation:
"""


class TestPipelines(unittest.TestCase):
    """
    API tests for pipelines
    """

    @staticmethod
    @patch.dict(os.environ, {"CONFIG": os.path.join(tempfile.gettempdir(), "testapi.yml"), "API_CLASS": "txtai.api.API"})
    def start():
        """
        Starts a mock FastAPI client.
        """

        config = os.path.join(tempfile.gettempdir(), "testapi.yml")

        with open(config, "w") as output:
            output.write(PIPELINES)

        client = TestClient(app)
        start()

        return client

    @classmethod
    def setUpClass(cls):
        """
        Create API client on creation of class.
        """

        cls.client = TestPipelines.start()

        cls.data = [
            "US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day",
        ]

        cls.text = (
            "Search is the base of many applications. Once data starts to pile up, users want to be able to find it. It’s the foundation "
            "of the internet and an ever-growing challenge that is never solved or done. The field of Natural Language Processing (NLP) is "
            "rapidly evolving with a number of new developments. Large-scale general language models are an exciting new capability "
            "allowing us to add amazing functionality quickly with limited compute and people. Innovation continues with new models "
            "and advancements coming in at what seems a weekly basis. This article introduces txtai, an AI-powered search engine "
            "that enables Natural Language Understanding (NLU) based search in any application."
        )

    def testEmpty(self):
        """
        Test empty API configuration
        """

        api = API({})

        self.assertIsNone(api.label("test", ["test"]))
        self.assertIsNone(api.pipeline("junk", "test"))

    def testLabel(self):
        """
        Test label via API
        """

        labels = self.client.post("label", json={"text": "this is the best sentence ever", "labels": ["positive", "negative"]}).json()

        self.assertEqual(labels[0]["id"], 0)

    def testLabelBatch(self):
        """
        Test batch label via API
        """

        labels = self.client.post(
            "batchlabel", json={"texts": ["this is the best sentence ever", "This is terrible"], "labels": ["positive", "negative"]}
        ).json()

        results = [l[0]["id"] for l in labels]
        self.assertEqual(results, [0, 1])

    def testSegment(self):
        """
        Test segmentation via API
        """

        text = self.client.get("segment?text=This is a test. And another test.").json()

        # Check array length is 2
        self.assertEqual(len(text), 2)

    def testSegmentBatch(self):
        """
        Test batch segmentation via API
        """

        text = "This is a test. And another test."
        texts = self.client.post("batchsegment", json=[text, text]).json()

        # Check array length is 2 and first element length is 2
        self.assertEqual(len(texts), 2)
        self.assertEqual(len(texts[0]), 2)

    def testSimilarity(self):
        """
        Test similarity via API
        """

        uid = self.client.post("similarity", json={"query": "feel good story", "texts": self.data}).json()[0]["id"]

        self.assertEqual(self.data[uid], self.data[4])

    def testSimilarityBatch(self):
        """
        Test batch similarity via API
        """

        results = self.client.post("batchsimilarity", json={"queries": ["feel good story", "climate change"], "texts": self.data}).json()

        uids = [result[0]["id"] for result in results]
        self.assertEqual(uids, [4, 1])

    def testSummary(self):
        """
        Test summary via API
        """

        summary = self.client.get("summary?text=%s&minlength=15&maxlength=15" % urllib.parse.quote(self.text)).json()
        self.assertEqual(summary, "txtai is an AI-powered search engine that")

    def testSummaryBatch(self):
        """
        Test batch summary via API
        """

        summaries = self.client.post("batchsummary", json={"texts": [self.text, self.text], "maxlength": 10}).json()
        self.assertEqual(len(summaries), 2)

    def testTextractor(self):
        """
        Test textractor via API
        """

        text = self.client.get("textract?file=%s" % Utils.PATH + "/article.pdf").json()

        # Check length of text is as expected
        self.assertEqual(len(text), 2301)

    def testTextractorBatch(self):
        """
        Test batch textractor via API
        """

        path = Utils.PATH + "/article.pdf"

        texts = self.client.post("batchtextract", json=[path, path]).json()
        self.assertEqual(len(texts), 2)

    def testTranscribe(self):
        """
        Test transcribe via API
        """

        text = self.client.get("transcribe?file=%s" % Utils.PATH + "/Make_huge_profits.wav").json()

        # Check length of text is as expected
        self.assertEqual(text, "Make huge profits without working make up to one hundred thousand dollars a day")

    def testTranscribeBatch(self):
        """
        Test batch transcribe via API
        """

        path = Utils.PATH + "/Make_huge_profits.wav"

        texts = self.client.post("batchtranscribe", json=[path, path]).json()
        self.assertEqual(len(texts), 2)

    def testTranslate(self):
        """
        Test translate via API
        """

        translation = self.client.get("translate?text=%s&target=es" % urllib.parse.quote("This is a test translation into Spanish")).json()
        self.assertEqual(translation, "Esta es una traducción de prueba al español")

    def testTranslateBatch(self):
        """
        Test batch translate via API
        """

        text = "This is a test translation into Spanish"
        translations = self.client.post("batchtranslate", json={"texts": [text, text], "target": "es"}).json()
        self.assertEqual(len(translations), 2)
