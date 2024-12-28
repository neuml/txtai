"""
Encoding module tests
"""

import base64
import os
import tempfile
import unittest
import urllib.parse

from io import BytesIO
from unittest.mock import patch

import msgpack
import numpy as np
import PIL

from fastapi.testclient import TestClient

from txtai.api import application
from txtai.api.responses import JSONEncoder

# pylint: disable = C0411
from utils import Utils

# Configuration for image storage
INDEX = """
# Allow indexing of documents
writable: %s

embeddings:
    defaults: False
    content: True
    objects: %s
"""


class TestEncoding(unittest.TestCase):
    """
    API tests for response encoding
    """

    @staticmethod
    @patch.dict(os.environ, {"CONFIG": os.path.join(tempfile.gettempdir(), "testapi.yml"), "API_CLASS": "txtai.api.API"})
    def start(yaml):
        """
        Starts a mock FastAPI client.

        Args:
            yaml: input configuration
        """

        config = os.path.join(tempfile.gettempdir(), "testapi.yml")

        with open(config, "w", encoding="utf-8") as output:
            output.write(yaml)

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

        cls.client = TestEncoding.start(INDEX % ("True", "image"))

    def testImages(self):
        """
        Test image encoding
        """

        with open(Utils.PATH + "/books.jpg", "rb") as f:
            self.client.post("addimage", data={"uid": 0}, files={"data": f})
            self.client.get("index")

        query = urllib.parse.quote_plus("select id, object from txtai limit 1")
        results = self.client.get(f"search?query={query}").json()

        # Test reading image
        self.assertIsInstance(PIL.Image.open(BytesIO(base64.b64decode(results[0]["object"]))), PIL.Image.Image)

    def testInvalidInputs(self):
        """
        Test invalid parameter inputs
        """

        response = self.client.post("addimage", data={"uid": [0, 1]}, files={"data": b"123"})
        self.assertEqual(response.status_code, 422)

        response = self.client.post("addobject", data={"uid": [0, 1]}, files={"data": b"123"})
        self.assertEqual(response.status_code, 422)

    def testInvalidJSON(self):
        """
        Test that invalid JSON raises an exception
        """

        with self.assertRaises(TypeError):
            JSONEncoder().encode(np.random.rand(1, 1))

    def testMessagePack(self):
        """
        Test message pack encoding
        """

        # Validate binary encoding
        results = self.client.get("count", headers={"Accept": "application/msgpack"}).content
        self.assertEqual(results, b"\x01")

        # Validate query result
        query = urllib.parse.quote_plus("select id, object from txtai limit 1")
        results = self.client.get(f"search?query={query}", headers={"Accept": "application/msgpack"}).content
        results = msgpack.unpackb(results)

        # Test reading image
        self.assertIsInstance(PIL.Image.open(BytesIO(results[0]["object"])), PIL.Image.Image)

    def testObjects(self):
        """
        Test object encoding
        """

        # Recreate model with standard object encoding
        self.client = TestEncoding.start(INDEX % ("True", "True"))

        # Test various formats
        self.client.post("addobject", data={"uid": "id0"}, files={"data": b"1234"})
        self.client.post("addobject", files={"data": b"ABC"})
        self.client.post("addobject", data={"uid": "id1", "field": "object"}, files={"data": b"A1234"})
        self.client.get("index")

        query = urllib.parse.quote_plus("select id, object from txtai where id = 'id0' limit 1")
        results = self.client.get(f"search?query={query}").json()
        self.assertEqual(base64.b64decode(results[0]["object"]), b"1234")

        # Test with messagepack encoding
        results = self.client.get(f"search?query={query}", headers={"Accept": "application/msgpack"}).content
        results = msgpack.unpackb(results)
        self.assertEqual(results[0]["object"], b"1234")

        count = self.client.get("count").json()
        self.assertEqual(count, 3)

    def testReadOnly(self):
        """
        Test read only indexes
        """

        # Recreate model with standard object encoding
        self.client = TestEncoding.start(INDEX % ("False", "True"))

        # Test errors raised for write operations
        with open(Utils.PATH + "/books.jpg", "rb") as f:
            response = self.client.post("addimage", data={"uid": 0}, files={"data": f})
            self.assertEqual(response.status_code, 403)

        self.assertEqual(self.client.post("addobject", data={"uid": 0}, files={"data": b"1234"}).status_code, 403)
