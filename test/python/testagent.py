"""
Agent module tests
"""

import os
import tempfile
import unittest

from unittest.mock import patch

from datetime import datetime

from smolagents import CodeAgent, PythonInterpreterTool

from txtai.agent import Agent
from txtai.embeddings import Embeddings


class TestAgent(unittest.TestCase):
    """
    Agent tests.
    """

    def testExecute(self):
        """
        Test executing main agent loop
        """

        agent = Agent(llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1)

        # Patch LLM to generate answer
        agent.process.model.llm = lambda *args, **kwargs: 'Action:\n{"name": "final_answer", "arguments": "Hi"}'

        self.assertEqual(agent("Hello"), "Hi")

    def testMethod(self):
        """
        Test agent process methods
        """

        agent = Agent(method="code", llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_iterations=1)
        self.assertIsInstance(agent.process, CodeAgent)

    def testToolsBasic(self):
        """
        Test adding basic function tools
        """

        class DateTime:
            """
            Date time instance
            """

            def __call__(self, iso):
                """
                Gets the current date and time

                Args:
                    iso: date will be converted to iso format if True

                Returns:
                    current date and time
                """

                return datetime.today().isoformat() if iso else datetime.today()

        today = {"name": "today", "description": "Gets the current date and time", "target": DateTime()}

        def current(iso: str) -> str:
            """
            Gets the current date and time

            Args:
                iso: date will be converted to iso format if True

            Returns:
                current date and time
            """

            return datetime.today().isoformat() if iso else datetime.today()

        agent = Agent(tools=[today, current, "websearch"], llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1)

        self.assertIsNotNone(agent)
        self.assertIsInstance(agent.tools["today"](True), str)
        self.assertIsInstance(agent.tools["current"](True), str)

    def testToolsEmbeddings(self):
        """
        Test adding Embeddings as a tool
        """

        embeddings = Embeddings()
        embeddings.index(["test"])

        # Generate temp file path and save
        index = os.path.join(tempfile.gettempdir(), "embeddings.agent")
        embeddings.save(index)

        embeddings1 = {
            "name": "embeddings1",
            "description": "Searches a test database",
            "target": embeddings,
        }

        embeddings2 = {"name": "embeddings2", "description": "Searches a test database", "path": index}

        agent = Agent(tools=[embeddings1, embeddings2], llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1)

        self.assertIsNotNone(agent)
        self.assertIsInstance(agent.tools["embeddings1"]("test"), list)

    # pylint: disable=C0115,C0116
    @patch("mcpadapt.core.MCPAdapt")
    def testToolsMCP(self, mcp):
        """
        Test adding a MCP tool collection
        """

        class MCPAdapt:
            def __init__(self, *args):
                self.args = args

            def tools(self):
                return [PythonInterpreterTool()]

        # Patch MCP adapter for testing
        mcp.side_effect = MCPAdapt

        agent = Agent(tools=["http://localhost:8000/mcp"], llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1)
        self.assertEqual(len(agent.tools), 2)
