"""
Authorization module tests
"""

import hashlib
import os
import tempfile
import unittest

from unittest.mock import patch

from fastapi.testclient import TestClient

from txtai.api import application


class TestAuthorization(unittest.TestCase):
    """
    API tests for token authorization.
    """

    @staticmethod
    @patch.dict(
        os.environ,
        {
            "CONFIG": os.path.join(tempfile.gettempdir(), "testapi.yml"),
            "DEPENDENCIES": "txtai.api.Authorization",
            "TOKEN": hashlib.sha256("token".encode("utf-8")).hexdigest(),
        },
    )
    def start():
        """
        Starts a mock FastAPI client.
        """

        config = os.path.join(tempfile.gettempdir(), "testapi.yml")

        with open(config, "w", encoding="utf-8") as output:
            output.write("embeddings:\n")

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

        cls.client = TestAuthorization.start()

    def testInvalid(self):
        """
        Test invalid authorization
        """

        response = self.client.get("search?query=test")
        self.assertEqual(response.status_code, 401)

        response = self.client.get("search?query=test", headers={"Authorization": "Bearer invalid"})
        self.assertEqual(response.status_code, 401)

    def testValid(self):
        """
        Test valid authorization
        """

        results = self.client.get("search?query=test", headers={"Authorization": "Bearer token"}).json()
        self.assertEqual(results, [])
