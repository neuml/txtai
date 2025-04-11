"""
LiteLLM module tests
"""

import json
import os
import time
import unittest
import uuid

from unittest.mock import patch

from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

from txtai.pipeline import LLM


class RequestHandler(BaseHTTPRequestHandler):
    """
    Test HTTP handler.
    """

    def do_POST(self):
        """
        POST request handler.
        """

        # Parse input headers
        length = int(self.headers["content-length"])
        data = json.loads(self.rfile.read(length))

        if data.get("stream"):
            # Mock streaming response
            content = "application/octet-stream"
            response = (
                "data: "
                + json.dumps(
                    {
                        "id": str(uuid.uuid4()),
                        "object": "chat.completion.chunk",
                        "created": int(time.time() * 1000),
                        "model": "test",
                        "choices": [{"id": 0, "delta": {"content": "blue"}}],
                    }
                )
                + "\n\ndata: [DONE]\n\n"
            )
        else:
            # Mock standard response
            content = "application/json"
            response = json.dumps(
                {
                    "id": str(uuid.uuid4()),
                    "object": "chat.completion",
                    "created": int(time.time() * 1000),
                    "model": "test",
                    "choices": [{"id": 0, "message": {"role": "assistant", "content": "blue"}, "finish_reason": "stop"}],
                }
            )

        # Encode response as bytes
        response = response.encode("utf-8")

        self.send_response(200)
        self.send_header("content-type", content)
        self.send_header("content-length", len(response))
        self.end_headers()

        self.wfile.write(response)
        self.wfile.flush()


class TestLiteLLM(unittest.TestCase):
    """
    LiteLLM tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create mock http server.
        """

        cls.httpd = HTTPServer(("127.0.0.1", 8000), RequestHandler)

        server = Thread(target=cls.httpd.serve_forever, daemon=True)
        server.start()

    @classmethod
    def tearDownClass(cls):
        """
        Shutdown mock http server.
        """

        cls.httpd.shutdown()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test"})
    def testGeneration(self):
        """
        Test generation with LiteLLM
        """

        # Test model generation with LiteLLM
        model = LLM("gpt-4o", api_base="http://127.0.0.1:8000")
        self.assertEqual(model("The sky is"), "blue")

        # Test default role
        self.assertEqual(model("The sky is", defaultrole="user"), "blue")

        # Test streaming
        self.assertEqual(" ".join(x for x in model("The sky is", stream=True)), "blue")

        # Test vision
        self.assertEqual(model.isvision(), False)
