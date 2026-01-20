"""
OpenCode module tests
"""

import json
import unittest

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

        # Mock response
        content = "application/json"
        response = json.dumps({"id": "0", "parts": [{"type": "text", "text": "blue"}]})

        # Encode response as bytes
        response = response.encode("utf-8")

        self.send_response(200)
        self.send_header("content-type", content)
        self.send_header("content-length", len(response))
        self.end_headers()

        self.wfile.write(response)
        self.wfile.flush()


class TestOpenCode(unittest.TestCase):
    """
    OpenCode tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create mock http server.
        """

        cls.httpd = HTTPServer(("127.0.0.1", 8005), RequestHandler)

        server = Thread(target=cls.httpd.serve_forever, daemon=True)
        server.start()

    @classmethod
    def tearDownClass(cls):
        """
        Shutdown mock http server.
        """

        cls.httpd.shutdown()

    def testGeneration(self):
        """
        Test generation with OpenCode
        """

        # Test model generation with LiteLLM
        model = LLM("opencode/big-pickle", url="http://127.0.0.1:8005")
        self.assertEqual(model("The sky is"), "blue")
