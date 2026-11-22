"""
OpenAI module tests
"""

import json
import time
import unittest
import uuid

from unittest.mock import patch

from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

from txtai.pipeline import LLM, OpenAI
from txtai.pipeline.llm import GenerationFactory


class RequestHandler(BaseHTTPRequestHandler):
    """
    Test HTTP handler.
    """

    # Last request received
    request = {}

    def do_POST(self):
        """
        POST request handler.
        """

        # Parse input headers
        length = int(self.headers["content-length"])
        data = json.loads(self.rfile.read(length))

        # Save request for inspection
        RequestHandler.request = {"path": self.path, "data": data}

        # Completions API returns text, chat completions API returns messages
        if self.path.endswith("/chat/completions"):
            choice = {"delta": {"content": "blue"}} if data.get("stream") else {"message": {"role": "assistant", "content": "blue"}}
        else:
            choice = {"text": "blue"}

        response = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion.chunk" if data.get("stream") else "chat.completion",
            "created": int(time.time()),
            "model": data["model"],
            "choices": [{"index": 0, "finish_reason": "stop", **choice}],
        }

        if data.get("stream"):
            # Mock streaming response with a trailing usage chunk that has no choices
            content = "text/event-stream"
            usage = {**response, "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            response = f"data: {json.dumps(response)}\n\ndata: {json.dumps(usage)}\n\ndata: [DONE]\n\n"
        else:
            # Mock standard response
            content = "application/json"
            response = json.dumps(response)

        # Encode response as bytes
        response = response.encode("utf-8")

        self.send_response(200)
        self.send_header("content-type", content)
        self.send_header("content-length", len(response))
        self.end_headers()

        self.wfile.write(response)
        self.wfile.flush()


class TestOpenAI(unittest.TestCase):
    """
    OpenAI tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create mock http server.
        """

        cls.httpd = HTTPServer(("127.0.0.1", 8007), RequestHandler)

        server = Thread(target=cls.httpd.serve_forever, daemon=True)
        server.start()

    @classmethod
    def tearDownClass(cls):
        """
        Shutdown mock http server.
        """

        cls.httpd.shutdown()

    @patch("txtai.pipeline.llm.openai.Download")
    def testFactory(self, download):
        """
        Test OpenAI method is derived when LiteLLM is not installed
        """

        # Emulate models not found on the HF Hub
        download.return_value.side_effect = OSError

        # LiteLLM is checked first when installed
        with patch("txtai.pipeline.llm.litellm.LiteLLM.ismodel", return_value=True):
            self.assertEqual(GenerationFactory.method("openai/gpt-4o", None), "litellm")

        # Emulate LiteLLM not installed
        with patch("txtai.pipeline.llm.litellm.LITELLM", False):
            self.assertEqual(GenerationFactory.method("openai/gpt-4o", None), "openai")
            self.assertEqual(GenerationFactory.method("gpt-4o", None), "transformers")

            model = LLM("openai/gpt-4o", api_key="test", base_url="http://127.0.0.1:8007/v1")
            self.assertIsInstance(model.generator, OpenAI)

        # Models on the HF Hub are not OpenAI API models
        download.return_value.side_effect = None
        download.return_value.return_value = "config.json"
        self.assertFalse(OpenAI.ismodel("openai/gpt-oss-20b"))

    def testGeneration(self):
        """
        Test generation with OpenAI
        """

        # Test model generation with OpenAI
        model = LLM("openai/gpt-4o", method="openai", api_key="test", base_url="http://127.0.0.1:8007/v1", temperature=0)
        self.assertEqual(model("The sky is"), "blue")

        # Test request parameters. Provider prefix is removed and unset parameters are not sent.
        data = RequestHandler.request["data"]
        self.assertEqual(data["model"], "gpt-4o")
        self.assertEqual(data["messages"], [{"role": "user", "content": "The sky is"}])
        self.assertEqual((data["max_tokens"], data["temperature"]), (512, 0))
        self.assertNotIn("stop", data)

        # Test max_completion_tokens replaces max_tokens
        model("The sky is", max_completion_tokens=100)
        self.assertEqual(RequestHandler.request["data"]["max_completion_tokens"], 100)
        self.assertNotIn("max_tokens", RequestHandler.request["data"])

        # Test raw prompt uses the completions API
        self.assertEqual(model("The sky is", defaultrole="prompt", stop=["."]), "blue")
        request = RequestHandler.request
        self.assertTrue(request["path"].endswith("/v1/completions"))
        self.assertEqual((request["data"]["prompt"], request["data"]["stop"]), ("The sky is", ["."]))

        # Test streaming
        self.assertEqual(" ".join(x for x in model("The sky is", stream=True)), "blue")

        # Test vision
        self.assertEqual(model.isvision(), False)

        # Test api_base alias and model names without a provider prefix
        model = LLM("qwen3:8b", method="openai", api_key="test", api_base="http://127.0.0.1:8007/v1")
        self.assertEqual(model("The sky is"), "blue")
        self.assertEqual(RequestHandler.request["data"]["model"], "qwen3:8b")

    def testNotInstalled(self):
        """
        Test OpenAI when the openai package is not installed
        """

        with patch("txtai.pipeline.llm.openai.OPENAI", False):
            self.assertFalse(OpenAI.ismodel("openai/gpt-4o"))

            with self.assertRaises(ImportError):
                OpenAI("openai/gpt-4o")
