"""
OpenAI module tests
"""

import json
import os
import unittest

from unittest.mock import patch

from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

import numpy as np

from txtai.vectors import VectorsFactory
from txtai.vectors.dense import OpenAI


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

        # Generate mock response
        response = {
            "object": "list",
            "data": [{"object": "embedding", "index": x, "embedding": [1.0] * 768} for x in range(len(data["input"]))],
            "model": data["model"],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
        response = json.dumps(response).encode("utf-8")

        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", len(response))
        self.end_headers()

        self.wfile.write(response)
        self.wfile.flush()


class TestOpenAI(unittest.TestCase):
    """
    OpenAI vectors tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Create mock http server.
        """

        cls.httpd = HTTPServer(("127.0.0.1", 8008), RequestHandler)

        server = Thread(target=cls.httpd.serve_forever, daemon=True)
        server.start()

    @classmethod
    def tearDownClass(cls):
        """
        Shutdown mock http server.
        """

        cls.httpd.shutdown()

    @patch("txtai.vectors.dense.openai.OpenAI.ishub", return_value=False)
    def testFactory(self, _):
        """
        Test OpenAI method is derived when LiteLLM is not installed
        """

        config = {"path": "openai/text-embedding-3-small"}

        # LiteLLM is checked first when installed
        with patch("txtai.vectors.dense.litellm.LiteLLM.ismodel", return_value=True):
            self.assertEqual(VectorsFactory.method(config), "litellm")

        # Emulate LiteLLM not installed
        with patch("txtai.vectors.dense.litellm.LITELLM", False):
            self.assertEqual(VectorsFactory.method(config), "openai")
            self.assertEqual(VectorsFactory.method({"path": "text-embedding-3-small"}), "transformers")

            model = VectorsFactory.create({**config, "vectors": {"api_key": "test", "base_url": "http://127.0.0.1:8008/v1"}}, None)
            self.assertIsInstance(model, OpenAI)

            # Emulate OpenAI not installed
            with patch("txtai.vectors.dense.openai.OPENAI", False):
                self.assertEqual(VectorsFactory.method(config), "transformers")

    def testIndex(self):
        """
        Test indexing with OpenAI vectors
        """

        # OpenAI vectors instance
        model = VectorsFactory.create(
            {
                "method": "openai",
                "path": "openai/text-embedding-3-small",
                "vectors": {"api_key": "test", "api_base": "http://127.0.0.1:8008/v1", "dimensions": 768},
            },
            None,
        )

        ids, dimension, batches, stream = model.index([(0, "test", None)])

        self.assertEqual(len(ids), 1)
        self.assertEqual(dimension, 768)
        self.assertEqual(batches, 1)
        self.assertIsNotNone(os.path.exists(stream))

        # Test shape of serialized embeddings
        with open(stream, "rb") as queue:
            self.assertEqual(np.load(queue).shape, (1, 768))

        # Test request parameters
        request = RequestHandler.request
        self.assertTrue(request["path"].endswith("/embeddings"))
        self.assertEqual(request["data"]["model"], "text-embedding-3-small")
        self.assertEqual(request["data"]["input"], ["test"])
        self.assertEqual(request["data"]["dimensions"], 768)

    def testNotInstalled(self):
        """
        Test error raised when the openai package is not installed
        """

        with patch("txtai.vectors.dense.openai.OPENAI", False):
            self.assertFalse(OpenAI.ismodel("openai/text-embedding-3-small"))

            with self.assertRaises(ImportError):
                VectorsFactory.create({"method": "openai", "path": "openai/text-embedding-3-small"}, None)
