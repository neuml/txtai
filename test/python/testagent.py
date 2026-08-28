"""
Agent module tests
"""

import os
import tempfile
import unittest

from threading import Thread
from unittest.mock import patch

from datetime import datetime

from smolagents import CodeAgent, PythonInterpreterTool

from txtai.agent import Agent
from txtai.agent.tool import BashTool, ToolFactory
from txtai.embeddings import Embeddings

# agents.md content
AGENTS = """
Basic instructions here
"""

# Sample skill.md content
SKILL = """---
name: hello
description: says hello world
---

Says hello world
"""


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

    def testInstructions(self):
        """
        Test loading an agents.md file
        """

        # Test loading instructions from file
        agents = os.path.join(tempfile.gettempdir(), "agents.md")
        with open(agents, "w", encoding="utf-8") as output:
            output.write(AGENTS)

        agent = Agent(instructions=agents, llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_iterations=1)
        agent.process.model.llm = lambda *args, **kwargs: 'Action:\n{"name": "final_answer", "arguments": "Hi"}'
        self.assertEqual(agent("Hello"), "Hi")

        # Test loading from memory
        agent = Agent(instructions=AGENTS, llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_iterations=1)
        agent.process.model.llm = lambda *args, **kwargs: 'Action:\n{"name": "final_answer", "arguments": "Hi"}'
        self.assertEqual(agent("Hello"), "Hi")

    def testMemory(self):
        """
        Test agent memory
        """

        agent = Agent(llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1, memory=5)

        # Patch LLM to generate answer
        agent.process.model.llm = lambda *args, **kwargs: 'Action:\n{"name": "final_answer", "arguments": "Hi"}'

        self.assertEqual(agent("Hello"), "Hi")
        self.assertEqual(agent("Hello"), "Hi")

        # Test that results are stored in shared memory
        self.assertEqual(len(agent.memory.get(None)), 2)

        # Test resetting shared memory
        self.assertEqual(agent("Hello", reset=True), "Hi")
        self.assertEqual(len(agent.memory.get(None)), 1)

        # Test session memory
        self.assertEqual(agent("Hello", session="session-0"), "Hi")
        self.assertEqual(len(agent.memory.get("session-0")), 1)

        # Test resetting session memory
        self.assertEqual(agent("Hello", session="session-0", reset=True), "Hi")
        self.assertEqual(len(agent.memory.get("session-0")), 1)
        self.assertEqual(len(agent.memory.get(None)), 1)

    def testMethod(self):
        """
        Test agent process methods
        """

        agent = Agent(method="code", llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_iterations=1)
        self.assertIsInstance(agent.process, CodeAgent)

    def testSkill(self):
        """
        Test running a skill from a skill.md file
        """

        skill = os.path.join(tempfile.gettempdir(), "skill.md")
        with open(skill, "w", encoding="utf-8") as output:
            output.write(SKILL)

        agent = Agent(tools=[skill], llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_iterations=1)

        self.assertIsInstance(agent.tools["hello"]("say hello"), str)

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

    def testToolsDefaults(self):
        """
        Test default toolkit tools
        """

        agent = Agent(tools=["defaults"], llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1)

        # Working directory
        work = tempfile.gettempdir()

        # Test file
        path = os.path.join(work, "agent_tools.txt")
        agent.tools["write"](path, "hello world")

        # Test default tools
        self.assertIsNotNone(agent.tools["bash"](["ls", work]))
        self.assertGreater(len(agent.tools["glob"](work)), 0)
        self.assertGreater(len(agent.tools["grep"]("world", "*")), 0)
        self.assertEqual(agent.tools["todowrite"]("plan"), "plan")

        agent.tools["edit"](path, "hello", "goodbye")
        self.assertEqual(agent.tools["read"](path), "goodbye world".strip())

        # Test dynamic default tool creation
        tool = ToolFactory.default("bash")
        self.assertEqual(tool.name, "bash")
        self.assertIsNot(tool, ToolFactory.default("bash"))

    def testToolsBashStdin(self):
        """
        Test default bash tool doesn't block reading stdin
        """

        tool = ToolFactory.default("bash")

        # Replace stdin with a pipe that never reaches EOF, simulating a long-lived
        # parent process (i.e. an API service) that keeps stdin open
        read, write = os.pipe()
        stdin = os.dup(0)

        result = {}

        try:
            os.dup2(read, 0)

            # `cat` with no file arguments reads stdin - run in a thread to avoid blocking the test suite
            thread = Thread(target=lambda: result.update(output=tool(["cat"])), daemon=True)
            thread.start()
            thread.join(timeout=30)
        finally:
            os.dup2(stdin, 0)
            for descriptor in (stdin, read, write):
                os.close(descriptor)

        # Command must return and not block forever reading stdin
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["output"], "")

    def testToolsBashTimeout(self):
        """
        Test bash tool commands are stopped once they exceed the timeout
        """

        tool = BashTool(allowed=["sleep"], timeout=1)

        self.assertIn("timed out", tool(["sleep", "30"]))

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
