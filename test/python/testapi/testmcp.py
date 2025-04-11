"""
Agent API module tests
"""

import os
import tempfile
import unittest

from unittest.mock import patch

from fastapi.testclient import TestClient

from txtai.api import application

# Configuration for agents
MCP = """
mcp: True
"""


# pylint: disable=R0904
class TestMCP(unittest.TestCase):
    """
    API tests for model context protocol (MCP)
    """

    @staticmethod
    @patch.dict(os.environ, {"CONFIG": os.path.join(tempfile.gettempdir(), "testapi.yml"), "API_CLASS": "txtai.api.API"})
    def start():
        """
        Starts a mock FastAPI client.
        """

        config = os.path.join(tempfile.gettempdir(), "testapi.yml")

        with open(config, "w", encoding="utf-8") as output:
            output.write(MCP)

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

        cls.client = TestMCP.start()

    def testMCP(self):
        """
        Test that application a /mcp route
        """

        self.assertTrue(any(route.path == "/mcp" for route in self.client.app.routes))
