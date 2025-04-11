"""
OpenAI API module tests
"""

import os
import tempfile
import unittest

from unittest.mock import patch

from fastapi.testclient import TestClient

from txtai.api import application

# pylint: disable = C0411
from utils import Utils

# API Configuration
CONFIG = """
# Enable OpenAI-compatible API
openai: True

# Allow indexing of documents
writable: True

# Agent configuration
agent:
    hello:
        max_iterations: 1

# Embeddings settings
embeddings:
    path: sentence-transformers/nli-mpnet-base-v2
    content: True        

# LLM configuration
llm:
    path: hf-internal-testing/tiny-random-LlamaForCausalLM

# Text segmentation
segmentation:

# Text to speech
texttospeech:

# Transcription
transcription:

# Workflow
workflow:
    echo:
        tasks:
            - task: console
"""


# pylint: disable=R0904
class TestOpenAI(unittest.TestCase):
    """
    Tests for OpenAI-compatible API endpoint for txtai.
    """

    @staticmethod
    @patch.dict(os.environ, {"CONFIG": os.path.join(tempfile.gettempdir(), "testopenai.yml"), "API_CLASS": "txtai.api.API"})
    def start():
        """
        Starts a mock FastAPI client.
        """

        config = os.path.join(tempfile.gettempdir(), "testopenai.yml")

        with open(config, "w", encoding="utf-8") as output:
            output.write(CONFIG)

        # Create new application and set on client
        application.app = application.create()
        client = TestClient(application.app)
        application.start()

        # Patch LLM to generate answer
        agent = application.get().agents["hello"]
        agent.process.model.llm = lambda *args, **kwargs: 'Action:\n{"name": "final_answer", "arguments": "Hi"}'

        return client

    @classmethod
    def setUpClass(cls):
        """
        Create API client on creation of class.
        """

        cls.client = TestOpenAI.start()

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

    def testChatAgent(self):
        """
        Test a chat completion with an agent
        """

        response = self.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello"}], "model": "hello"}).json()

        self.assertEqual(response["choices"][0]["message"]["content"], "Hi")

    def testChatLLM(self):
        """
        Test a chat completion with a LLM
        """

        response = self.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello"}], "model": "llm"}).json()

        self.assertIsNotNone(response["choices"][0]["message"]["content"])

    def testChatPipeline(self):
        """
        Test a chat completion with a pipeline
        """

        response = self.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello"}], "model": "segmentation"}).json()

        self.assertEqual(response["choices"][0]["message"]["content"], "Hello")

    def testChatSearch(self):
        """
        Test a chat completion with an embeddings search
        """

        response = self.client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "feel good story"}], "model": "embeddings"}
        ).json()

        self.assertEqual(response["choices"][0]["message"]["content"], self.data[4])

    def testChatStream(self):
        """
        Test a chat completion with a LLM
        """

        response = self.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello"}], "model": "llm", "stream": True})

        self.assertGreater(len(response.text.split("\n\n")), 0)

    def testChatWorkflow(self):
        """
        Test a chat completion with a workflow
        """

        response = self.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello"}], "model": "echo"}).json()

        self.assertEqual(response["choices"][0]["message"]["content"], "Hello")

    def testEmbeddings(self):
        """
        Test generating embeddings vectors
        """

        response = self.client.post("/v1/embeddings", json={"input": "text to embed", "model": "nli-mpnet-base-v2"}).json()

        self.assertEqual(len(response["data"][0]["embedding"]), 768)

    def testSpeech(self):
        """
        Test generating speech for input text
        """

        response = self.client.post(
            "/v1/audio/speech", json={"model": "tts", "input": "text to speak", "voice": "default", "response_format": "wav"}
        ).content

        self.assertTrue(response[0:4] == b"RIFF")

    def testTranscribe(self):
        """
        Test audio to text transcription
        """

        path = Utils.PATH + "/Make_huge_profits.wav"
        with open(path, "rb") as f:
            text = self.client.post("/v1/audio/transcriptions", files={"file": f}).json()["text"]
            self.assertEqual(text, "Make huge profits without working make up to one hundred thousand dollars a day")

    def testTranslate(self):
        """
        Test audio translation
        """

        path = Utils.PATH + "/Make_huge_profits.wav"
        with open(path, "rb") as f:
            text = self.client.post("/v1/audio/translations", files={"file": f}).json()["text"]
            self.assertEqual(text, "Make huge profits without working make up to one hundred thousand dollars a day")
