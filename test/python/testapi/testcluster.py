"""
Cluster API module tests
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

# Configuration for an embeddings cluster
CLUSTER = """
cluster:
    shards:
        - http://127.0.0.1:8002
        - http://127.0.0.1:8003
"""


class RequestHandler(BaseHTTPRequestHandler):
    """
    Test HTTP handler
    """

    def do_GET(self):
        """
        GET request handler.
        """

        if self.path == "/count":
            response = 5
        elif self.path.startswith("/search"):
            response = [{"id": 4, "score": 0.40}]
        else:
            response = {"result": "ok"}

        # Convert response to string
        response = json.dumps(response).encode("utf-8")

        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", len(response))
        self.end_headers()

        self.wfile.write(response)

    def do_POST(self):
        """
        POST request handler.
        """

        if self.path.startswith("/batchsearch"):
            response = [[{"id": 4, "score": 0.40}], [{"id": 1, "score": 0.40}]]
        elif self.path.startswith("/delete"):
            response = [0]
        else:
            response = {"result": "ok"}

        response = json.dumps(response).encode("utf-8")

        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", len(response))
        self.end_headers()

        self.wfile.write(response)


class TestCluster(unittest.TestCase):
    """
    API tests for embeddings clusters
    """

    @staticmethod
    @patch.dict(os.environ, {"CONFIG": os.path.join(tempfile.gettempdir(), "testapi.yml"), "API_CLASS": "txtai.api.API"})
    def start():
        """
        Starts a mock FastAPI client.
        """

        config = os.path.join(tempfile.gettempdir(), "testapi.yml")

        with open(config, "w") as output:
            output.write(CLUSTER)

        client = TestClient(app)
        start()

        return client

    @classmethod
    def setUpClass(cls):
        """
        Create API client on creation of class.
        """

        cls.client = TestCluster.start()

        cls.httpd1 = HTTPServer(("127.0.0.1", 8002), RequestHandler)

        server1 = Thread(target=cls.httpd1.serve_forever)
        server1.setDaemon(True)
        server1.start()

        cls.httpd2 = HTTPServer(("127.0.0.1", 8003), RequestHandler)

        server2 = Thread(target=cls.httpd2.serve_forever)
        server2.setDaemon(True)
        server2.start()

        # Index data
        cls.client.post("add", json=[{"id": 0, "text": "test"}])
        cls.client.get("index")

    @classmethod
    def tearDownClass(cls):
        """
        Shutdown mock http server.
        """

        cls.httpd1.shutdown()
        cls.httpd2.shutdown()

    def testCount(self):
        """
        Test cluster count
        """

        self.assertEqual(self.client.get("count").json(), 10)

    def testDelete(self):
        """
        Test cluster delete
        """

        self.assertEqual(self.client.post("delete", json=[0]).json(), [0])

    def testDeleteString(self):
        """
        Test string id
        """

        self.assertEqual(self.client.post("delete", json=["0"]).json(), [0])

    def testSearch(self):
        """
        Test cluster search
        """

        uid = self.client.get("search?query=feel%20good%20story&limit=1").json()[0]["id"]
        self.assertEqual(uid, 4)

    def testSearchBatch(self):
        """
        Test cluster batch search
        """

        results = self.client.post("batchsearch", json={"queries": ["feel good story", "climate change"], "limit": 1}).json()

        uids = [result[0]["id"] for result in results]
        self.assertEqual(uids, [4, 1])

    def testUpsert(self):
        """
        Test cluster upsert
        """

        # Update data
        self.client.post("add", json=[{"id": 4, "text": "Feel good story: baby panda born"}])
        self.client.get("upsert")

        # Search for best match
        uid = self.client.get("search?query=feel%20good%20story&limit=1").json()[0]["id"]

        self.assertEqual(uid, 4)
