"""
Agent API module tests
"""

import os
import tempfile
import unittest

from unittest.mock import patch

from fastapi.testclient import TestClient

from txtai.api import API, application

# Configuration for agents
AGENTS = """
agent:
    test:
        max_iterations: 1
        tools:
            - name: testtool
              description: Test tool
              target: testapi.testapiagent.TestTool

llm:
    path: hf-internal-testing/tiny-random-LlamaForCausalLM
"""


# pylint: disable=R0904
class TestAgent(unittest.TestCase):
    """
    API tests for agents.
    """

    @staticmethod
    @patch.dict(os.environ, {"CONFIG": os.path.join(tempfile.gettempdir(), "testapi.yml"), "API_CLASS": "txtai.api.API"})
    def start():
        """
        Starts a mock FastAPI client.
        """

        config = os.path.join(tempfile.gettempdir(), "testapi.yml")

        with open(config, "w", encoding="utf-8") as output:
            output.write(AGENTS)

        # Create new application and set on client
        application.app = application.create()
        client = TestClient(application.app)
        application.start()

        # Patch LLM to generate answer
        agent = application.get().agents["test"]
        agent.process.model.llm = lambda *args, **kwargs: 'Action:\n{"name": "final_answer", "arguments": "Hi"}'

        return client

    @classmethod
    def setUpClass(cls):
        """
        Create API client on creation of class.
        """

        cls.client = TestAgent.start()

    def testAgent(self):
        """
        Test agent via API
        """

        results = self.client.post("agent", json={"name": "test", "text": "Hello"}).json()
        self.assertEqual(results, "Hi")

    def testEmpty(self):
        """
        Test empty API configuration
        """

        api = API({})

        self.assertIsNone(api.agent("junk", "test"))


class TestTool:
    """
    Class to test agent tools
    """

    def __call__(self):
        pass
