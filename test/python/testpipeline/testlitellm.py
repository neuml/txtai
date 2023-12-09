"""
LiteLLM module tests
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

        # Generate mock response
        response = [{"generated_text": "blue"}]
        response = json.dumps(response).encode("utf-8")

        self.send_response(200)
        self.send_header("content-type", "application/json")
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

        server = Thread(target=cls.httpd.serve_forever)
        server.setDaemon(True)
        server.start()

    @classmethod
    def tearDownClass(cls):
        """
        Shutdown mock http server.
        """

        cls.httpd.shutdown()

    def testGeneration(self):
        """
        Test generation with LiteLLM
        """

        # Test model generation with llama.cpp
        model = LLM("huggingface/t5-small", api_base="http://127.0.0.1:8000")
        self.assertEqual(model("The sky is"), "blue")
