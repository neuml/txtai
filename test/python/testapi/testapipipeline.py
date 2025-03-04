"""
Pipeline API module tests
"""

import os
import tempfile
import unittest
import urllib

from unittest.mock import patch

from fastapi.testclient import TestClient

from txtai.api import API, application

# pylint: disable = C0411
from utils import Utils

# Configuration for pipelines
PIPELINES = """
# Image captions
caption:

# Entity extraction
entity:
    path: dslim/bert-base-NER

# Extractor settings
extractor:
    similarity: similarity
    path: llm

# Label settings
labels:
    path: prajjwal1/bert-medium-mnli

# LLM settings
llm:
    path: hf-internal-testing/tiny-random-gpt2
    task: language-generation

# Image objects
objects:

# Text segmentation
segmentation:
    sentences: true

# Enable pipeline similarity backed by zero shot classifier
similarity:

# Summarization
summary:
    path: t5-small

# Tabular
tabular:

# Text extraction
textractor:

# Text to speech
texttospeech:

# Transcription
transcription:

# Translation:
translation:

# Enable file uploads
upload:
"""


# pylint: disable=R0904
class TestPipeline(unittest.TestCase):
    """
    API tests for pipelines.
    """

    @staticmethod
    @patch.dict(os.environ, {"CONFIG": os.path.join(tempfile.gettempdir(), "testapi.yml"), "API_CLASS": "txtai.api.API"})
    def start():
        """
        Starts a mock FastAPI client.
        """

        config = os.path.join(tempfile.gettempdir(), "testapi.yml")

        with open(config, "w", encoding="utf-8") as output:
            output.write(PIPELINES)

        # Create new application and set on client
        application.app = application.create()
        client = TestClient(application.app)
        application.start()

        return client

    @classmethod
    def setUpClass(cls):
        """
        Create API client on creation of class.
        """

        cls.client = TestPipeline.start()

        cls.data = [
            "US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day",
        ]

        cls.text = (
            "Search is the base of many applications. Once data starts to pile up, users want to be able to find it. It's the foundation "
            "of the internet and an ever-growing challenge that is never solved or done. The field of Natural Language Processing (NLP) is "
            "rapidly evolving with a number of new developments. Large-scale general language models are an exciting new capability "
            "allowing us to add amazing functionality quickly with limited compute and people. Innovation continues with new models "
            "and advancements coming in at what seems a weekly basis. This article introduces txtai, an AI-powered search engine "
            "that enables Natural Language Understanding (NLU) based search in any application."
        )

    def testCaption(self):
        """
        Test caption via API
        """

        caption = self.client.get(f"caption?file={Utils.PATH}/books.jpg").json()

        self.assertEqual(caption, "a book shelf filled with books and a stack of books")

    def testCaptionBatch(self):
        """
        Test batch caption via API
        """

        path = Utils.PATH + "/books.jpg"

        captions = self.client.post("batchcaption", json=[path, path]).json()
        self.assertEqual(captions, ["a book shelf filled with books and a stack of books"] * 2)

    def testEntity(self):
        """
        Test entity extraction via API
        """

        entities = self.client.get(f"entity?text={self.data[1]}").json()
        self.assertEqual([e[0] for e in entities], ["Canada", "Manhattan"])

    def testEntityBatch(self):
        """
        Test batch entity via API
        """

        entities = self.client.post("batchentity", json=[self.data[1]]).json()
        self.assertEqual([e[0] for e in entities[0]], ["Canada", "Manhattan"])

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

    def testLLM(self):
        """
        Test LLM inference via API
        """

        response = self.client.get("llm?text=test").json()
        self.assertIsInstance(response, str)

    def testLLMBatch(self):
        """
        Test batch LLM inference via API
        """

        response = self.client.post("batchllm", json={"texts": ["test", "test"]}).json()
        self.assertEqual(len(response), 2)

    def testObjects(self):
        """
        Test objects via API
        """

        objects = self.client.get(f"objects?file={Utils.PATH}/books.jpg").json()

        self.assertEqual(objects[0][0], "book")

    def testObjectsBatch(self):
        """
        Test batch objects via API
        """

        path = Utils.PATH + "/books.jpg"

        objects = self.client.post("batchobjects", json=[path, path]).json()
        self.assertEqual([o[0][0] for o in objects], ["book"] * 2)

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

        summary = self.client.get(f"summary?text={urllib.parse.quote(self.text)}&minlength=15&maxlength=15").json()
        self.assertEqual(summary, "the field of natural language processing (NLP) is rapidly evolving")

    def testSummaryBatch(self):
        """
        Test batch summary via API
        """

        summaries = self.client.post("batchsummary", json={"texts": [self.text, self.text], "minlength": 15, "maxlength": 15}).json()
        self.assertEqual(summaries, ["the field of natural language processing (NLP) is rapidly evolving"] * 2)

    def testTabular(self):
        """
        Test tabular via API
        """

        results = self.client.get(f"tabular?file={Utils.PATH}/tabular.csv").json()

        # Check length of results is as expected
        self.assertEqual(len(results), 6)

    def testTabularBatch(self):
        """
        Test batch tabular via API
        """

        path = Utils.PATH + "/tabular.csv"

        results = self.client.post("batchtabular", json=[path, path]).json()
        self.assertEqual((len(results[0]), len(results[1])), (6, 6))

    def testTextractor(self):
        """
        Test textractor via API
        """

        text = self.client.get(f"textract?file={Utils.PATH}/article.pdf").json()

        # Check length of text is as expected
        self.assertEqual(len(text), 2471)

    def testTextractorBatch(self):
        """
        Test batch textractor via API
        """

        path = Utils.PATH + "/article.pdf"

        texts = self.client.post("batchtextract", json=[path, path]).json()
        self.assertEqual((len(texts[0]), len(texts[1])), (2471, 2471))

    def testTextToSpeech(self):
        """
        Test text to speech
        """

        # Generate audio and check for WAV signature
        audio = self.client.get("texttospeech?text=hello&encoding=wav").content
        self.assertTrue(audio[0:4] == b"RIFF")

    def testTranscribe(self):
        """
        Test transcribe via API
        """

        text = self.client.get(f"transcribe?file={Utils.PATH}/Make_huge_profits.wav").json()

        # Check length of text is as expected
        self.assertEqual(text, "Make huge profits without working make up to one hundred thousand dollars a day")

    def testTranscribeBatch(self):
        """
        Test batch transcribe via API
        """

        path = Utils.PATH + "/Make_huge_profits.wav"

        texts = self.client.post("batchtranscribe", json=[path, path]).json()
        self.assertEqual(texts, ["Make huge profits without working make up to one hundred thousand dollars a day"] * 2)

    def testTranslate(self):
        """
        Test translate via API
        """

        translation = self.client.get(f"translate?text={urllib.parse.quote('This is a test translation into Spanish')}&target=es").json()
        self.assertEqual(translation, "Esta es una traducción de prueba al español")

    def testTranslateBatch(self):
        """
        Test batch translate via API
        """

        text = "This is a test translation into Spanish"
        translations = self.client.post("batchtranslate", json={"texts": [text, text], "target": "es"}).json()
        self.assertEqual(translations, ["Esta es una traducción de prueba al español"] * 2)

    def testUpload(self):
        """
        Test file upload
        """

        path = Utils.PATH + "/article.pdf"
        with open(path, "rb") as f:
            path = self.client.post("upload", files={"files": f}).json()[0]
            self.assertTrue(os.path.exists(path))
