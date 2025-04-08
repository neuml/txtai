"""
Workflow API module tests
"""

import json
import os
import tempfile
import unittest

from http.server import HTTPServer, BaseHTTPRequestHandler
from multiprocessing.pool import ThreadPool
from threading import Thread
from unittest.mock import patch

from fastapi.testclient import TestClient

from txtai.api import API, application

# Configuration for workflows
WORKFLOWS = """
# Embeddings index
writable: true
embeddings:
    path: sentence-transformers/nli-mpnet-base-v2

# Labels
labels:
    path: prajjwal1/bert-medium-mnli

nop:

# Text segmentation
segmentation:
    sentences: true

# Workflow definitions
workflow:
    labels:
        tasks:
            - action: labels
              args: [[positive, negative]]
    multiaction:
        tasks:
            - action:
                - labels
                - nop
              initialize: testapi.testapiworkflow.TestInitFinal
              finalize: testapi.testapiworkflow.TestInitFinal
              merge: concat
              args:
                - [[positive, negative], false, True]
                - null
    schedule:
        schedule:
            cron: '* * * * * *'
            elements:
                - This is a test sentence. And another sentence to split.
            iterations: 1
        tasks:
            - action: segmentation
    segment:
        tasks:
            - action: segmentation
            - action: index
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
              url: http://127.0.0.1:8001/testpost
              params:

    xml:
        tasks:
            - task: service
              url: http://127.0.0.1:8001/xml
              method: get
              batch: false
              extract: row
              params:
                text:
"""


class RequestHandler(BaseHTTPRequestHandler):
    """
    Test HTTP handler.
    """

    def do_GET(self):
        """
        GET request handler.
        """

        self.send_response(200)

        if self.path.startswith("/xml"):
            response = "<row><text>test</text></row>".encode("utf-8")
            mime = "application/xml"
        else:
            response = '[{"text": "test"}]'.encode("utf-8")
            mime = "application/json"

        self.send_header("content-type", mime)
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
    API tests for workflows.
    """

    @staticmethod
    @patch.dict(os.environ, {"CONFIG": os.path.join(tempfile.gettempdir(), "testapi.yml"), "API_CLASS": "txtai.api.API"})
    def start():
        """
        Starts a mock FastAPI client.
        """

        config = os.path.join(tempfile.gettempdir(), "testapi.yml")

        with open(config, "w", encoding="utf-8") as output:
            output.write(WORKFLOWS)

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

        cls.client = TestWorkflow.start()

        cls.httpd = HTTPServer(("127.0.0.1", 8001), RequestHandler)

        server = Thread(target=cls.httpd.serve_forever, daemon=True)
        server.start()

    @classmethod
    def tearDownClass(cls):
        """
        Shutdown mock http server.
        """

        cls.httpd.shutdown()

    def testAPICleanup(self):
        """
        Test API threadpool closed when __del__ called.
        """

        api = API({})
        api.pool = ThreadPool()

        # pylint: disable=C2801
        api.__del__()

        self.assertIsNone(api.pool)

    def testServiceGet(self):
        """
        Test workflow with ServiceTask GET via API
        """

        text = "This is a test sentence. And another sentence to split."
        results = self.client.post("workflow", json={"name": "get", "elements": [text]}).json()

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 1)

    def testServicePost(self):
        """
        Test workflow with ServiceTask POST via API
        """

        text = "This is a test sentence. And another sentence to split."
        results = self.client.post("workflow", json={"name": "post", "elements": [text]}).json()

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 2)

    def testServiceXml(self):
        """
        Test workflow with ServiceTask GET via API and XML response
        """

        text = "This is a test sentence. And another sentence to split."
        results = self.client.post("workflow", json={"name": "xml", "elements": [text]}).json()

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 1)

    def testWorkflowLabels(self):
        """
        Test workflow with labels via API
        """

        text = "This is the best"

        results = self.client.post("workflow", json={"name": "labels", "elements": [text]}).json()
        self.assertEqual(results[0][0], 0)

        results = self.client.post("workflow", json={"name": "multiaction", "elements": [text]}).json()
        self.assertEqual(results[0], "['positive']. This is the best")

    def testWorkflowSegment(self):
        """
        Test workflow with segmentation via API
        """

        text = "This is a test sentence. And another sentence to split."

        results = self.client.post("workflow", json={"name": "segment", "elements": [text]}).json()
        self.assertEqual(len(results), 2)

        results = self.client.post("workflow", json={"name": "segment", "elements": [[0, text]]}).json()
        self.assertEqual(len(results), 2)


class TestInitFinal:
    """
    Class to test task initialize and finalize calls.
    """

    def __call__(self):
        pass
