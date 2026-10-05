"""
Dependency module tests
"""

import os
import tempfile
import unittest

from unittest.mock import patch

from fastapi.testclient import TestClient

from txtai.api import application


class SampleDependency:
    """
    Sample dependency
    """

    def __call__(self):
        raise NotImplementedError("Dependency test")


class TestDependency(unittest.TestCase):
    """
    API tests for dependency injection
    """

    @staticmethod
    @patch.dict(
        os.environ,
        {"CONFIG": os.path.join(tempfile.gettempdir(), "testapi.yml"), "DEPENDENCIES": "testapi.testdependency.SampleDependency"},
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

        cls.client = TestDependency.start()

    def testDependency(self):
        """
        Test dependency included
        """

        with self.assertRaises(NotImplementedError):
            self.client.get("search?query=test")
