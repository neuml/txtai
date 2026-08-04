"""
Agent module tests
"""

import os
import subprocess
import sys
import tempfile
import unittest

from unittest.mock import patch

from datetime import datetime

from smolagents import CodeAgent, PythonInterpreterTool, Tool

from txtai.agent import Agent
from txtai.agent.tool import ToolFactory
from txtai.agent.tool.glob import GlobTool
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


class TestToolFactory(unittest.TestCase):
    """
    ToolFactory tests.
    """

    def testDefaultsAreConstructors(self):
        """
        Test the default toolkit stores constructors and not instances
        """

        # Instances built in the class body run at import time. A tool needing an optional
        # dependency would then raise while txtai.agent is importing, which agent/__init__.py
        # turns into the placeholder Agent and a message naming the wrong package.
        for name, constructor in ToolFactory.DEFAULTS.items():
            self.assertTrue(isinstance(constructor, type), f"{name} is not a constructor")
            self.assertTrue(issubclass(constructor, Tool), f"{name} is not a Tool")

    def testDefaultCreatesTool(self):
        """
        Test resolving a default tool by alias name
        """

        tool = ToolFactory.default("bash")

        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "bash")

    def testDefaultCaches(self):
        """
        Test default tool instances are created once and reused
        """

        self.assertIs(ToolFactory.default("glob"), ToolFactory.default("glob"))

    def testDefaultCachesAliases(self):
        """
        Test an alias and its backwards compatible mapping share a single instance
        """

        # webview is an alias for read
        self.assertIs(ToolFactory.DEFAULTS["webview"], ToolFactory.DEFAULTS["read"])

    def testDefaultUnknownName(self):
        """
        Test an unknown default tool name raises
        """

        with self.assertRaises(KeyError):
            ToolFactory.default("notatool")

    def testDefaultDeferred(self):
        """
        Test a default tool is not constructed until it is requested
        """

        calls = []

        class Counted(Tool):
            """
            Tool that records each construction
            """

            name = "counted"
            description = "Counts constructions"
            inputs = {}
            output_type = "any"

            def __init__(self):
                calls.append(1)
                super().__init__()

            # pylint: disable=W0221
            def forward(self):
                return len(calls)

        with patch.dict(ToolFactory.DEFAULTS, {"counted": Counted}), patch.dict(ToolFactory.INSTANCES, {}, clear=True):
            # Registering the tool must not construct it
            self.assertEqual(len(calls), 0)

            # Requesting it constructs once, then reuses the instance
            ToolFactory.default("counted")
            ToolFactory.default("counted")
            self.assertEqual(len(calls), 1)

    def testDefaultRaisesWhenRequested(self):
        """
        Test a broken default tool only raises when that tool is requested
        """

        class Broken(Tool):
            """
            Tool standing in for one whose optional dependency is missing
            """

            name = "broken"
            description = "Always fails to build"
            inputs = {}
            output_type = "any"

            # pylint: disable=W0231
            def __init__(self):
                raise ImportError('example pipeline is not available - install "pipeline" extra to enable')

            # pylint: disable=W0221
            def forward(self):
                return None

        with patch.dict(ToolFactory.DEFAULTS, {"broken": Broken}), patch.dict(ToolFactory.INSTANCES, {}, clear=True):
            # Other tools still resolve
            self.assertIsInstance(ToolFactory.default("bash"), Tool)

            # The failure surfaces only for the requested tool, naming the extra that is missing
            with self.assertRaises(ImportError) as context:
                ToolFactory.default("broken")

            self.assertIn("pipeline", str(context.exception))

    def testCreateDefaultAlias(self):
        """
        Test create resolves a default tool alias
        """

        tools = ToolFactory.create({"tools": ["bash"]})

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "bash")

    def testCreateDefaultsToolkit(self):
        """
        Test create resolves the default toolkit without duplicates
        """

        # Small toolkit with an alias, mirroring webview -> read
        toolkit = {"glob": GlobTool, "files": GlobTool}

        with patch.dict(ToolFactory.DEFAULTS, toolkit, clear=True), patch.dict(ToolFactory.INSTANCES, {}, clear=True):
            tools = ToolFactory.create({"tools": ["defaults"]})

            # Both names map to one constructor, so the toolkit holds a single instance
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].name, "glob")

    def testCreateEmpty(self):
        """
        Test create with no tools configured
        """

        self.assertEqual(ToolFactory.create({}), [])

    def testImportWithMissingExtra(self):
        """
        Test a default tool with a missing optional dependency doesn't disable agents
        """

        # Run in a subprocess to get a clean import of txtai.agent. bs4 backs the Textractor the
        # read tool builds and ships in the "pipeline" extra, so blocking it stands in for an
        # install that only has the documented "agent" extra.
        program = """
import sys

# Block the "pipeline" extra dependency backing the read tool
sys.modules["bs4"] = None

from txtai.agent import Agent

print(Agent.__module__)
"""

        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=False)

        # Agents must still be available, and must not fall back to the placeholder stub
        self.assertEqual(result.stdout.strip().splitlines()[-1], "txtai.agent.base", result.stderr)
