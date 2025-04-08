"""
Cluster API module tests
"""

import json
import os
import tempfile
import unittest
import urllib.parse

from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from unittest.mock import patch

from fastapi.testclient import TestClient

from txtai.api import application

# Configuration for an embeddings cluster
CLUSTER = """
cluster:
    shards:
        - http://127.0.0.1:8002
        - http://127.0.0.1:8003
"""


class RequestHandler(BaseHTTPRequestHandler):
    """
    Test HTTP handler.
    """

    def do_GET(self):
        """
        GET request handler.
        """

        if self.path == "/count":
            response = 26
        elif self.path.startswith("/search?query=select"):
            if "group+by+id" in self.path:
                response = [{"count(*)": 26}]
            elif "group+by+text" in self.path:
                response = [{"count(*)": 12, "text": "This is a test"}, {"count(*)": 14, "text": "And another test"}]
            elif "group+by+txt" in self.path:
                response = [{"count(*)": 12, "txt": "This is a test"}, {"count(*)": 14, "txt": "And another test"}]
            else:
                if self.server.server_port == 8002:
                    response = [{"count(*)": 12, "min(indexid)": 0, "max(indexid)": 11, "avg(indexid)": 6.3}]
                else:
                    response = [{"count(*)": 16, "min(indexid)": 2, "max(indexid)": 14, "avg(indexid)": 6.7}]
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
        self.wfile.flush()

    def do_POST(self):
        """
        POST request handler.
        """

        if self.path.startswith("/batchsearch"):
            response = [[{"id": 4, "score": 0.40}], [{"id": 1, "score": 0.40}]]
        elif self.path.startswith("/delete"):
            if self.server.server_port == 8002:
                response = [0]
            else:
                response = []
        else:
            response = {"result": "ok"}

        response = json.dumps(response).encode("utf-8")

        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", len(response))
        self.end_headers()

        self.wfile.write(response)
        self.wfile.flush()


@unittest.skipIf(os.name == "nt", "TestCluster skipped on Windows")
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

        with open(config, "w", encoding="utf-8") as output:
            output.write(CLUSTER)

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

        cls.client = TestCluster.start()

        cls.httpd1 = HTTPServer(("127.0.0.1", 8002), RequestHandler)

        server1 = Thread(target=cls.httpd1.serve_forever, daemon=True)
        server1.start()

        cls.httpd2 = HTTPServer(("127.0.0.1", 8003), RequestHandler)

        server2 = Thread(target=cls.httpd2.serve_forever, daemon=True)
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

        self.assertEqual(self.client.get("count").json(), 52)

    def testDelete(self):
        """
        Test cluster delete
        """

        self.assertEqual(self.client.post("delete", json=[0]).json(), [0])

    def testDeleteString(self):
        """
        Test cluster delete with string id
        """

        self.assertEqual(self.client.post("delete", json=["0"]).json(), [0])

    def testIds(self):
        """
        Test id configurations
        """

        # String ids
        self.client.post("add", json=[{"id": "0", "text": "test"}])
        self.assertEqual(self.client.get("index").status_code, 200)

        # Auto ids
        self.client.post("add", json=[{"text": "test"}])
        self.assertEqual(self.client.get("index").status_code, 200)

    def testReindex(self):
        """
        Test cluster reindex
        """

        self.assertEqual(self.client.post("reindex", json={"config": {"path": "sentence-transformers/nli-mpnet-base-v2"}}).status_code, 200)

    def testSearch(self):
        """
        Test cluster search
        """

        # Encode parameters
        params = json.dumps({"x": 1})

        query = urllib.parse.quote("feel good story")
        uid = self.client.get(f"search?query={query}&limit=1&weights=0.5&index=default&parameters={params}&graph=False").json()[0]["id"]
        self.assertEqual(uid, 4)

    def testSearchBatch(self):
        """
        Test cluster batch search
        """

        results = self.client.post(
            "batchsearch",
            json={
                "queries": ["feel good story", "climate change"],
                "limit": 1,
                "weights": 0.5,
                "index": "default",
                "parameters": [{"x": 1}, {"x": 2}],
                "graph": False,
            },
        ).json()

        uids = [result[0]["id"] for result in results]
        self.assertEqual(uids, [4, 1])

    def testSQL(self):
        """
        Test cluster SQL statement
        """

        query = urllib.parse.quote("select count(*), min(indexid), max(indexid), avg(indexid) from txtai where text='This is a test'")
        self.assertEqual(
            self.client.get(f"search?query={query}").json(), [{"count(*)": 28, "min(indexid)": 0, "max(indexid)": 14, "avg(indexid)": 6.5}]
        )

        query = urllib.parse.quote("select count(*), text txt from txtai group by txt order by count(*) desc")
        self.assertEqual(
            self.client.get(f"search?query={query}").json(),
            [{"count(*)": 28, "txt": "And another test"}, {"count(*)": 24, "txt": "This is a test"}],
        )

        query = urllib.parse.quote("select count(*), text from txtai group by text order by count(*) asc")
        self.assertEqual(
            self.client.get(f"search?query={query}").json(),
            [{"count(*)": 24, "text": "This is a test"}, {"count(*)": 28, "text": "And another test"}],
        )

        query = urllib.parse.quote("select count(*) from txtai group by id order by count(*)")
        self.assertEqual(self.client.get(f"search?query={query}").json(), [{"count(*)": 52}])

    def testUpsert(self):
        """
        Test cluster upsert
        """

        # Update data
        self.client.post("add", json=[{"id": 4, "text": "Feel good story: baby panda born"}])
        self.client.get("upsert")

        # Search for best match
        query = "feel good story"
        uid = self.client.get(f"search?query={query}&limit=1").json()[0]["id"]

        self.assertEqual(uid, 4)
