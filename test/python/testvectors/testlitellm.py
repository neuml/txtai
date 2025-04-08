"""
LiteLLM module tests
"""

import json
import os
import unittest

from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

import numpy as np

from txtai.vectors import VectorsFactory


class RequestHandler(BaseHTTPRequestHandler):
    """
    Test HTTP handler.
    """

    def do_POST(self):
        """
        POST request handler.
        """

        # Generate mock response
        response = [[0.0] * 768]
        response = json.dumps(response).encode("utf-8")

        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", len(response))
        self.end_headers()

        self.wfile.write(response)
        self.wfile.flush()


class TestLiteLLM(unittest.TestCase):
    """
    LiteLLM vectors tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Create mock http server.
        """

        cls.httpd = HTTPServer(("127.0.0.1", 8004), RequestHandler)

        server = Thread(target=cls.httpd.serve_forever, daemon=True)
        server.start()

    @classmethod
    def tearDownClass(cls):
        """
        Shutdown mock http server.
        """

        cls.httpd.shutdown()

    def testIndex(self):
        """
        Test indexing with LiteLLM vectors
        """

        # LiteLLM vectors instance
        model = VectorsFactory.create(
            {"path": "huggingface/sentence-transformers/all-MiniLM-L6-v2", "vectors": {"api_base": "http://127.0.0.1:8004"}}, None
        )

        ids, dimension, batches, stream = model.index([(0, "test", None)])

        self.assertEqual(len(ids), 1)
        self.assertEqual(dimension, 768)
        self.assertEqual(batches, 1)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (1, 768))
