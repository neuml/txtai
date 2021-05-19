"""
Workflow API module tests
"""

import json
import os
import tempfile
import unittest

from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from unittest.mock import patch

from fastapi.testclient import TestClient

from txtai.api import app, start

# Configuration for workflows
WORKFLOWS = """
# Text segmentation
segmentation:
    sentences: true

# Workflow definitions
workflow:
    segment:
        tasks:
            - action: segmentation
    get:
        tasks:
            - task: service
              url: http://127.0.0.1:8001/testget
              method: get
              params:
                text:
    post:
        tasks:
            - task: service
              url: http://127.0.0.1:8001/batchsegment
              params:
"""


class RequestHandler(BaseHTTPRequestHandler):
    """
    Test HTTP handler
    """

    def do_GET(self):
        """
        GET request handler.
        """

        response = '[{"text": "test"}]'.encode("utf-8")

        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", len(response))
        self.end_headers()

        self.wfile.write(response)
        self.wfile.flush()

    def do_POST(self):
        """
        POST request handler.
        """

        length = int(self.headers["content-length"])
        data = json.loads(self.rfile.read(length))

        response = json.dumps([[y for y in x.split(".") if y] for x in data]).encode("utf-8")

        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", len(response))
        self.end_headers()

        self.wfile.write(response)
        self.wfile.flush()


class TestWorkflow(unittest.TestCase):
    """
    API tests for workflows
    """

    @staticmethod
    @patch.dict(os.environ, {"CONFIG": os.path.join(tempfile.gettempdir(), "testapi.yml"), "API_CLASS": "txtai.api.API"})
    def start():
        """
        Starts a mock FastAPI client.
        """

        config = os.path.join(tempfile.gettempdir(), "testapi.yml")

        with open(config, "w") as output:
            output.write(WORKFLOWS)

        client = TestClient(app)
        start()

        return client

    @classmethod
    def setUpClass(cls):
        """
        Create API client on creation of class.
        """

        cls.client = TestWorkflow.start()

        cls.httpd = HTTPServer(("127.0.0.1", 8001), RequestHandler)

        server = Thread(target=cls.httpd.serve_forever)
        server.setDaemon(True)
        server.start()

    @classmethod
    def tearDownClass(cls):
        """
        Shutdown mock http server.
        """

        cls.httpd.shutdown()

    def testServiceGet(self):
        """
        Test workflow with ServiceTask GET via API.
        """

        text = "This is a test sentence. And another sentence to split."
        results = self.client.post("workflow", json={"name": "get", "elements": [text]}).json()

        self.assertEqual(len(results), 1)

    def testServicePost(self):
        """
        Test workflow with ServiceTask POST via API.
        """

        text = "This is a test sentence. And another sentence to split."
        results = self.client.post("workflow", json={"name": "post", "elements": [text]}).json()

        self.assertEqual(len(results), 2)

    def testWorkflow(self):
        """
        Test workflow via API.
        """

        text = "This is a test sentence. And another sentence to split."
        results = self.client.post("workflow", json={"name": "segment", "elements": [text]}).json()
        self.assertEqual(len(results), 2)
