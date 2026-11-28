"""
Agent module tests
"""

import copy
import os
import sys
import tempfile
import traceback
import unittest
import warnings

from threading import Thread
from unittest.mock import patch

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Tuple, Union

import torch

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from smolagents import CodeAgent, PythonInterpreterTool, ToolCallingAgent, tool as smoltool

from txtai.agent import Agent
from txtai.agent.pydanticai import PydanticProcess
from txtai.agent.pydanticai.tool import PydanticTool
from txtai.agent.smolagents import SmolagentsTool
from txtai.agent.tool import BashTool, DocstringError, QuestionTool, Tool, ToolFactory, ToolSchema, TypeHintError, WebSearchTool
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

    def testBackend(self):
        """
        Test agent backend selection
        """

        # Default backend is smolagents
        agent = Agent(llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1)
        self.assertIsInstance(agent.process, ToolCallingAgent)

        agent = Agent(backend="smolagents", llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1)
        self.assertIsInstance(agent.process, ToolCallingAgent)

        agent = Agent(backend="pydantic", llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1)
        self.assertIsInstance(agent.process, PydanticProcess)

        # Unknown backends raise an error
        with self.assertRaises(ImportError):
            Agent(backend="unknown", llm="hf-internal-testing/tiny-random-LlamaForCausalLM")

    def testCustomBackend(self):
        """
        Test resolving a custom backend
        """

        # Built-in process runner resolved by class path
        agent = Agent(backend="txtai.agent.pydanticai.PydanticProcess", llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1)
        self.assertIsInstance(agent.process, PydanticProcess)

        # Custom backends are created with the model, txtai tools and remaining options
        agent = Agent(backend="testagent.CustomProcess", llm="hf-internal-testing/tiny-random-LlamaForCausalLM", tools=["read"], option=1)
        self.assertIsInstance(agent.process, CustomProcess)
        self.assertEqual(agent.process.llm, "hf-internal-testing/tiny-random-LlamaForCausalLM")
        self.assertIsInstance(agent.tools["read"], Tool)
        self.assertEqual(agent.process.config, {"option": 1})
        self.assertEqual(agent("Hello"), "Hello")

    def testCustomBackendInvalid(self):
        """
        Test resolving an invalid backend
        """

        for backend in ["pprint.pprint", "txtai.agent.Missing"]:
            with self.subTest(backend=backend), self.assertRaises(ImportError):
                Agent(backend=backend, llm="hf-internal-testing/tiny-random-LlamaForCausalLM")

    def testExecute(self):
        """
        Test executing main agent loop
        """

        agent = Agent(llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1)

        # Patch LLM to generate answer
        agent.process.model.llm = lambda *args, **kwargs: 'Action:\n{"name": "final_answer", "arguments": "Hi"}'

        self.assertEqual(agent("Hello"), "Hi")

    def testImportError(self):
        """
        Test missing and transitive dependencies preserve their import failure.
        """

        # pylint: disable=C0415
        for module in ("jinja2.sandbox",):
            with self.subTest(module=module), patch.dict(sys.modules, {module: None}):
                # Reload txtai with one dependency unavailable, as in testoptional.
                for name in list(sys.modules):
                    if name == "txtai" or name.startswith("txtai."):
                        del sys.modules[name]

                from txtai import Agent as UnavailableAgent

                with self.assertRaises(ImportError) as error:
                    UnavailableAgent()

                cause = error.exception.__cause__
                self.assertIsInstance(cause, ModuleNotFoundError)
                self.assertIs(cause, UnavailableAgent.ERROR)
                self.assertEqual(cause.name, module)
                self.assertIsNotNone(cause.__traceback__)
                self.assertIn(str(cause), "".join(traceback.format_exception(error.exception)))
                self.assertIn('install "agent" extra', str(error.exception))

            # Module restoration must keep the real Agent available to other tests.
            from txtai import Agent as RestoredAgent

            self.assertIs(RestoredAgent, Agent)

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

    def testOptionalSmolagents(self):
        """
        Test agent tools and the pydantic backend work without smolagents installed
        """

        def weather(city: str) -> str:
            """
            Gets the weather for a city

            Args:
                city: city name

            Returns:
                weather
            """

            return f"sunny in {city}"

        # pylint: disable=C0415
        with patch.dict(sys.modules, {"smolagents": None}):
            # Reload txtai with smolagents unavailable, as in testoptional
            for name in list(sys.modules):
                if name == "txtai" or name.startswith("txtai."):
                    del sys.modules[name]

            from txtai import agent as optional

            # Agent and tools import without smolagents
            OptionalAgent, OptionalToolFactory = optional.Agent, optional.ToolFactory
            self.assertIsNone(getattr(OptionalAgent, "ERROR", None))
            self.assertIsNotNone(optional.BashTool(allowed=["ls"])({"command": ["ls", tempfile.gettempdir()]}))

            # Tools only available with the smolagents backend raise an error
            self.assertNotIn("python", OptionalToolFactory.DEFAULTS)
            for tool in ["python", "http://localhost:8000/mcp"]:
                with self.subTest(tool=tool), self.assertRaises(ValueError):
                    OptionalToolFactory.create({"tools": [tool]})

            # The smolagents backend raises an error
            with self.assertRaises(ImportError) as error:
                OptionalAgent(llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1)

            self.assertIn('install "agent" extra', str(error.exception))

            # The pydantic backend runs default and function tools
            agent = OptionalAgent(
                backend="pydantic", tools=[weather, "defaults"], llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=5
            )
            self.assertEqual(
                sorted(agent.tools), ["bash", "edit", "glob", "grep", "read", "todowrite", "user_input", "weather", "web_search", "write"]
            )

            responses = iter(['Action:\n{"name": "weather", "arguments": {"city": "Paris"}}', "Done"])
            messages = []
            agent.process.model.llm = lambda inputs, **kwargs: messages.append(inputs) or next(responses)

            self.assertEqual(agent("What is the weather in Paris?"), "Done")
            self.assertEqual(messages[1][-1], {"role": "user", "content": "Observation: sunny in Paris"})

        # Module restoration must keep the real Agent available to other tests
        from txtai import Agent as RestoredAgent

        self.assertIs(RestoredAgent, Agent)

    def testPydantic(self):
        """
        Test executing main agent loop with the pydantic backend
        """

        agent = Agent(backend="pydantic", llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1)

        # Test plain text answer
        agent.process.model.llm = lambda *args, **kwargs: "Hi"
        self.assertEqual(agent("Hello"), "Hi")

        # Test final_answer action format
        agent.process.model.llm = lambda *args, **kwargs: 'Action:\n{"name": "final_answer", "arguments": "Hi"}'
        self.assertEqual(agent("Hello"), "Hi")

        agent.process.model.llm = lambda *args, **kwargs: 'Action:\n{"name": "final_answer", "arguments": {"answer": "Hi"}}'
        self.assertEqual(agent("Hello"), "Hi")

        # Test streaming
        self.assertEqual(list(agent("Hello", stream=True)), ["Hi"])

    def testPydanticConfig(self):
        """
        Test pydantic backend configuration
        """

        # Test loading model from a dictionary and custom instructions
        agent = Agent(
            backend="pydantic", model={"path": "hf-internal-testing/tiny-random-LlamaForCausalLM"}, instructions="Be brief", max_iterations=1
        )

        messages = []
        agent.process.model.llm = lambda inputs, **kwargs: messages.extend(inputs) or "Hi"

        self.assertEqual(agent("Hello"), "Hi")
        self.assertEqual(messages[0], {"role": "system", "content": "Be brief"})
        self.assertEqual(messages[1]["role"], "user")

        # Test maxlength is passed to the LLM
        agent.process.model.llm = lambda inputs, **kwargs: str(kwargs["maxlength"])
        self.assertEqual(agent("Hello", maxlength=1024), "1024")

        # Code agents aren't supported
        with self.assertRaises(ValueError):
            Agent(backend="pydantic", method="code", llm="hf-internal-testing/tiny-random-LlamaForCausalLM")

    def testPydanticImportError(self):
        """
        Test the pydantic backend raises an error when Pydantic AI isn't installed
        """

        # pylint: disable=C0415
        with patch.dict(sys.modules, {"pydantic_ai": None}):
            # Reload txtai with pydantic_ai unavailable, as in testoptional
            for name in list(sys.modules):
                if name == "txtai" or name.startswith("txtai."):
                    del sys.modules[name]

            from txtai import Agent as UnavailableAgent

            with self.assertRaises(ImportError) as error:
                UnavailableAgent(backend="pydantic", llm="hf-internal-testing/tiny-random-LlamaForCausalLM")

            self.assertIn('install "agent" extra', str(error.exception))

            # smolagents backend still works without pydantic_ai
            self.assertIsNotNone(UnavailableAgent(llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1))

        # Module restoration must keep the real Agent available to other tests
        from txtai import Agent as RestoredAgent

        self.assertIs(RestoredAgent, Agent)

    def testPydanticMemory(self):
        """
        Test agent memory with the pydantic backend
        """

        agent = Agent(backend="pydantic", llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1, memory=5)

        messages = []
        agent.process.model.llm = lambda inputs, **kwargs: messages.append(inputs) or "Hi"

        self.assertEqual(agent("Hello"), "Hi")
        self.assertEqual(agent("Goodbye"), "Hi")
        self.assertEqual(len(agent.memory.get(None)), 2)

        # Prior requests are added to the prompt
        self.assertIn("User: Hello\nAssistant: Hi", messages[-1][-1]["content"])

    def testPydanticMessages(self):
        """
        Test converting Pydantic AI messages to chat messages and parsing LLM responses
        """

        agent = Agent(backend="pydantic", llm="hf-internal-testing/tiny-random-LlamaForCausalLM")
        model = agent.process.model

        # RetryPromptPart is deprecated in newer versions of Pydantic AI but still supported
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            retry = RetryPromptPart(content="Invalid arguments", tool_name="search")

        messages = model.messages(
            [
                ModelRequest(parts=[SystemPromptPart("Be brief"), SystemPromptPart("Use tools"), UserPromptPart(["Hello", "World"])]),
                ModelResponse(parts=[TextPart("Hi")]),
                ModelRequest(parts=[retry]),
                ModelResponse(parts=[ToolCallPart(tool_name="search", args={"query": "txtai"}, tool_call_id="1")]),
                ModelRequest(parts=[ToolReturnPart(tool_name="search", content="results", tool_call_id="1")]),
            ],
            ModelRequestParameters(),
        )

        # Consecutive messages with the same role are merged
        self.assertEqual(messages[0], {"role": "system", "content": "Be brief\n\nUse tools"})
        self.assertEqual(messages[1], {"role": "user", "content": "Hello\nWorld"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "Hi"})
        self.assertEqual(messages[3]["role"], "user")
        self.assertIn("Invalid arguments", messages[3]["content"])
        self.assertEqual(messages[4], {"role": "assistant", "content": 'Action:\n{"name": "search", "arguments": {"query": "txtai"}}'})
        self.assertEqual(messages[5], {"role": "user", "content": "Observation: results"})

        # Tool calls that aren't JSON objects are sent back to the model to retry
        part = model.parse('Action:\n["search"]')
        self.assertEqual((part.tool_name, part.args), ("", '["search"]'))

    def testPydanticOutput(self):
        """
        Test structured output with the pydantic backend
        """

        agent = Agent(backend="pydantic", llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=5, output_type=int)

        # LLM that answers with plain text, then calls the final result tool
        responses = iter(["Five", 'Action:\n{"name": "final_result", "arguments": {"response": 5}}'])

        messages = []
        agent.process.model.llm = lambda inputs, **kwargs: messages.append(inputs) or next(responses)

        self.assertEqual(agent("Pick a number"), 5)

        # Final result tools are described in the system prompt
        self.assertIn("call one of the final result tools", messages[0][0]["content"])
        self.assertIn("- final_result:", messages[0][0]["content"])

        # Plain text answers are sent back to the LLM with validation feedback
        self.assertEqual(messages[1][2], {"role": "assistant", "content": "Five"})
        self.assertEqual(messages[1][3]["role"], "user")

    def testPydanticSteps(self):
        """
        Test the pydantic backend stops once max steps is reached
        """

        def search(query: str) -> str:
            """
            Runs a search

            Args:
                query: search query

            Returns:
                search results
            """

            return query

        agent = Agent(backend="pydantic", tools=[search], llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=2)

        # LLM that never stops calling tools
        agent.process.model.llm = lambda *args, **kwargs: 'Action:\n{"name": "search", "arguments": {"query": "txtai"}}'

        with self.assertRaises(UsageLimitExceeded):
            agent("Hello")

    def testPydanticTools(self):
        """
        Test tool calling with the pydantic backend
        """

        def weather(city: str) -> str:
            """
            Gets the weather for a city

            Args:
                city: city name

            Returns:
                weather
            """

            return f"sunny in {city}"

        agent = Agent(backend="pydantic", tools=[weather, "defaults"], llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=5)

        # Tools dictionary has the original tools
        self.assertEqual(agent.tools["weather"]("Paris"), "sunny in Paris")
        self.assertIn("bash", agent.tools)

        # LLM that calls a tool, sends an invalid tool call, then answers with the tool result
        responses = iter(
            [
                'Action:\n{"name": "weather", "arguments": {"city": "Paris"}}\nObservation: rainy',
                "Action:\n{invalid",
                "Done",
            ]
        )

        messages = []
        agent.process.model.llm = lambda inputs, **kwargs: messages.append(inputs) or next(responses)

        self.assertEqual(agent("What is the weather in Paris?"), "Done")

        # Tools are described in the system prompt
        self.assertIn("- weather: Gets the weather for a city", messages[0][0]["content"])
        self.assertIn('"city": {"type": "string", "description": "city name"}', messages[0][0]["content"])

        # Tool call and result are sent back to the LLM, generated observations are ignored
        self.assertEqual(messages[1][-2], {"role": "assistant", "content": 'Action:\n{"name": "weather", "arguments": {"city": "Paris"}}'})
        self.assertEqual(messages[1][-1], {"role": "user", "content": "Observation: sunny in Paris"})

        # Invalid tool calls are sent back to the LLM to retry
        self.assertEqual(messages[2][-1]["role"], "user")
        self.assertIn("Unknown tool name", messages[2][-1]["content"])

    def testPydanticToolSchema(self):
        """
        Test building a JSON schema from tool inputs
        """

        schema = PydanticTool.schema(
            {
                "query": {"type": "string", "description": "search query"},
                "limit": {"type": "integer", "description": "max results", "nullable": True},
                "data": {"type": "any", "description": "any data"},
            }
        )

        self.assertEqual(
            schema,
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                    "limit": {"type": "integer", "description": "max results"},
                    "data": {"description": "any data"},
                },
                "required": ["query", "data"],
            },
        )

    def testSkill(self):
        """
        Test running a skill from a skill.md file
        """

        skill = os.path.join(tempfile.gettempdir(), "skill.md")
        with open(skill, "w", encoding="utf-8") as output:
            output.write(SKILL)

        agent = Agent(tools=[skill], llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_iterations=1)

        self.assertIsInstance(agent.tools["hello"]("say hello"), str)

    def testToolsBase(self):
        """
        Test the txtai tool base class
        """

        class Echo(Tool):
            """
            Echo tool
            """

            def __init__(self):
                self.name = "echo"
                self.description = "Echoes text"
                self.inputs = {"text": {"type": "string", "description": "text"}}
                self.output = "string"
                super().__init__()

            # pylint: disable=W0221
            def forward(self, text):
                return text

        # Inputs can be passed as positional, keyword or a single dictionary of named inputs
        tool = Echo()
        self.assertEqual(tool("a"), "a")
        self.assertEqual(tool(text="b"), "b")
        self.assertEqual(tool({"text": "c"}), "c")

        # Dictionaries that aren't named inputs are passed through
        self.assertEqual(tool({"other": 1}), {"other": 1})

        # Tools must set all parameters
        # pylint: disable=W0223
        class Missing(Tool):
            """
            Tool missing a description
            """

            def __init__(self):
                self.name = "missing"
                self.inputs = {}
                self.output = "any"
                super().__init__()

        with self.assertRaises(TypeError):
            Missing()

        # Forward must be implemented
        tool = Tool.__new__(Tool)
        with self.assertRaises(NotImplementedError):
            tool.forward()

    def testToolsBashAllowedEmpty(self):
        """
        Test an empty allowed commands list permits no commands
        """

        # An empty allow list is an explicit choice and must be honored, not replaced by the defaults
        self.assertEqual(BashTool(allowed=[]).allowed, [])

        # No command runs when nothing is allowed
        self.assertIsNone(BashTool(allowed=[])(["cat", "file.txt"]))

        # The defaults still apply when allowed isn't set, including an explicit None
        self.assertEqual(BashTool().allowed, ["cat", "cut", "diff", "grep", "head", "ls", "tail"])
        self.assertEqual(BashTool(allowed=None).allowed, ["cat", "cut", "diff", "grep", "head", "ls", "tail"])

    @patch("subprocess.run")
    def testToolsBashEncoding(self, run):
        """
        Test bash tool decodes subprocess output as UTF-8 instead of the locale encoding
        """

        run.return_value.stdout = "output"

        tool = BashTool(allowed=["cat"])
        tool(["cat", "file.txt"])

        # Subprocess output must be decoded as UTF-8, not left to the locale's default encoding
        self.assertEqual(run.call_args.kwargs.get("encoding"), "utf-8")

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

        # Timeout is disabled when set to None
        self.assertIsNone(BashTool(timeout=None).timeout)

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

    def testToolsConfig(self):
        """
        Test function tool name and description from a dictionary configuration
        """

        def search(query: str) -> str:
            """
            Runs a search

            Args:
                query: search query

            Returns:
                search results
            """

            return query

        tool = ToolFactory.create({"tools": [{"name": "websearcher", "description": "Searches the web", "target": search}]})[0]
        self.assertEqual(tool.name, "websearcher")
        self.assertEqual(tool.description, "Searches the web")
        self.assertEqual(tool("txtai"), "txtai")

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

    def testToolsDocstring(self):
        """
        Test function tools without complete type hints are created from documentation
        """

        def untyped(city):
            """
            Gets the weather for a city

            Args:
                city: city name
            """

            return f"sunny in {city}"

        def noreturn(city: str):
            """
            Gets the weather for a city

            Args:
                city: city name
            """

            return f"sunny in {city}"

        for function in [untyped, noreturn]:
            tool = ToolFactory.create({"tools": [function]})[0]
            self.assertEqual(tool.name, function.__name__)
            self.assertEqual(tool("Paris"), "sunny in Paris")

    def testToolsEditCRLF(self):
        """
        Test the edit tool preserves CRLF line endings
        """

        # Test file with CRLF line endings
        path = os.path.join(tempfile.gettempdir(), "agent_tools_crlf.txt")
        with open(path, "wb") as f:
            f.write(b"hello\r\nworld\r\n")

        # Editing a word must not rewrite the line endings of untouched lines
        ToolFactory.default("edit")(path, "hello", "bye")
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"bye\r\nworld\r\n")

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

    def testToolsGrepUnreadable(self):
        """
        Test the grep tool skips a file it is not allowed to read
        """

        with tempfile.TemporaryDirectory() as work:
            for name in ("a.txt", "b.txt"):
                with open(os.path.join(work, name), "w", encoding="utf-8") as f:
                    f.write("hello world\n")

            # Matches found in readable files must survive an unreadable file
            unreadable = os.path.join(work, "b.txt")
            os.chmod(unreadable, 0o000)

            try:
                if os.access(unreadable, os.R_OK):
                    self.skipTest("File permissions are not enforced for this user")

                result = ToolFactory.default("grep")("world", work)
                self.assertEqual(result, {os.path.join(work, "a.txt"): ["hello world\n"]})
            finally:
                os.chmod(unreadable, 0o644)

    def testToolsHints(self):
        """
        Test tools created from type hints match the inputs smolagents creates
        """

        # pylint: disable=W0613
        def search(query: str, limit: int = 5, tags: list[str] = None, data=None, scale: Optional[float] = None) -> list:
            """
            Runs a search

            Args:
                query: search query
                limit: maximum results
                tags: tags to filter on
                data: extra data
                scale: optional score scale

            Returns:
                search results
            """

            return [query]

        # pylint: disable=W0613
        def anydata(query: str, data: Any = None, limit: int = 5, scale: Optional[float] = None, ids: Union[list[int], dict] = None) -> str:
            """
            Runs a search

            Args:
                query: search query
                data: extra data
                limit: maximum results
                scale: optional score scale
                ids: ids to filter on

            Returns:
                search results
            """

            return query

        def noinputs() -> int:
            """
            Returns a number
            """

            return 1

        def noreturn():
            """
            Does nothing
            """

        def anyoutput(query: str) -> Any:
            """
            Runs a search

            Args:
                query: search query

            Returns:
                search results
            """

            return query

        def unionoutput(query: str) -> Union[int, str]:
            """
            Runs a search

            Args:
                query: search query

            Returns:
                search results
            """

            return query

        for function in [anydata, noinputs, noreturn, anyoutput, unionoutput]:
            with self.subTest(function=function.__name__):
                tool, expected = ToolFactory.create({"tools": [function]})[0], smoltool(function)
                self.assertIsInstance(tool, Tool)
                self.assertEqual(
                    (tool.name, tool.description, tool.output),
                    (expected.name, expected.description, expected.output_type),
                )

                # Input keys are compared in order since they are rendered into prompts
                self.assertEqual([list(x.items()) for x in tool.inputs.values()], [list(x.items()) for x in expected.inputs.values()])
                self.assertEqual(list(tool.inputs), list(expected.inputs))

        # Inputs missing a type hint fall back to the docstring
        tool = ToolFactory.create({"tools": [search]})[0]
        self.assertEqual(tool.inputs, {"query": {"type": "any", "description": "search query"}})
        self.assertEqual(tool("txtai"), ["txtai"])

    @patch("mcpadapt.core.MCPAdapt")
    def testToolsMCP(self, mcp):
        """
        Test adding a MCP tool collection
        """

        # pylint: disable=C0115,C0116
        class MCPAdapt:
            def __init__(self, *args):
                self.args = args

            def tools(self):
                return [PythonInterpreterTool()]

        # Patch MCP adapter for testing
        mcp.side_effect = MCPAdapt

        agent = Agent(tools=["http://localhost:8000/mcp"], llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1)
        self.assertEqual(len(agent.tools), 2)

        # MCP tools require the smolagents backend
        with self.assertRaises(ValueError):
            Agent(backend="pydantic", tools=["http://localhost:8000/mcp"], llm="hf-internal-testing/tiny-random-LlamaForCausalLM")

    def testToolsQuestion(self):
        """
        Test the question tool asks the user
        """

        tool = ToolFactory.create({"tools": ["question"]})[0]
        self.assertIsInstance(tool, QuestionTool)
        self.assertEqual(tool.name, "user_input")

        with patch("builtins.input", return_value="Paris") as prompt:
            self.assertEqual(tool("Which city?"), "Paris")
            prompt.assert_called_once_with("Which city? => Type your answer here:")

    def testToolsReadNonexistent(self):
        """
        Test the read tool raises FileNotFoundError when a file does not exist
        """

        # Reading a non-existent file must raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            ToolFactory.default("read")(os.path.join(tempfile.gettempdir(), "does_not_exist_file.txt"))

    def testToolsSchema(self):
        """
        Test JSON schemas built from type hints and docstrings
        """

        class Color(Enum):
            """
            Colors
            """

            RED = "red"
            BLUE = "blue"

        # pylint: disable=W0613,R0917
        def search(
            query: str,
            mode: Literal["fast", 1],
            color: Color,
            pair: tuple[int, str],
            scores: dict[str, float],
            kind: str = "web",
            limit: int | None = None,
        ) -> list[str]:
            """
            Runs a search

            Args:
                query: search query
                    spanning two lines
                mode: search mode
                color: color filter
                pair: id and label
                scores: scores by id
                kind (str): search kind (choices: ["web", "news"])
                limit: maximum results

            Returns:
                search results
            """

            return [query]

        self.assertEqual(
            ToolSchema.create(search),
            {
                "name": "search",
                "description": "Runs a search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "search query spanning two lines"},
                        "mode": {"type": ["string", "integer"], "enum": ["fast", 1], "description": "search mode"},
                        "color": {"type": "string", "enum": ["red", "blue"], "description": "color filter"},
                        "pair": {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}], "description": "id and label"},
                        "scores": {"type": "object", "additionalProperties": {"type": "number"}, "description": "scores by id"},
                        "kind": {"type": "string", "enum": ["web", "news"], "description": "search kind"},
                        "limit": {"type": "integer", "nullable": True, "description": "maximum results"},
                    },
                    "required": ["query", "mode", "color", "pair", "scores"],
                },
                "return": {"type": "array", "items": {"type": "string"}, "description": "search results"},
            },
        )

        class Search:
            """
            Search instance
            """

            def run(self, query: str) -> object:
                """
                Runs a search

                Args:
                    query: search query
                """

                return query

        # The receiver is skipped and a return type that doesn't convert is left out
        self.assertEqual(
            ToolSchema.create(Search().run)["parameters"],
            {"type": "object", "properties": {"query": {"type": "string", "description": "search query"}}, "required": ["query"]},
        )

        # pylint: disable=W0613
        def nodoc(query: str) -> str:
            return query

        def undescribed(query: str, limit: int) -> str:
            """
            Runs a search

            Args:
                query: search query
            """

            return query

        class Custom:
            """
            Custom class
            """

        def custom(query: Custom) -> str:
            """
            Runs a search

            Args:
                query: search query
            """

            return query

        def single(query: tuple[str]) -> str:
            """
            Runs a search

            Args:
                query: search query
            """

            return query

        for function, error in [(nodoc, DocstringError), (undescribed, DocstringError), (custom, TypeHintError), (single, TypeHintError)]:
            with self.subTest(function=function.__name__), self.assertRaises(error):
                ToolSchema.create(function)

        # Bare tuples are arrays and image and audio types are matched by name
        self.assertEqual(ToolSchema.hint(Tuple), {"type": "array"})
        self.assertEqual(ToolSchema.hint(torch.Tensor), {"type": "audio"})

        class Empty(Enum):
            """
            Enum without members
            """

        class Pair(Enum):
            """
            Enum with a value that isn't a JSON scalar
            """

            PAIR = (1, 2)

        for hint in [Literal[b"fast"], set[int], Empty, Pair]:
            with self.subTest(hint=hint), self.assertRaises(TypeHintError):
                ToolSchema.hint(hint)

    def testToolsSmolagents(self):
        """
        Test txtai tools are converted to smolagents tools with the smolagents backend
        """

        agent = Agent(tools=["bash", PythonInterpreterTool()], llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=1)

        # txtai tools are wrapped, smolagents tools are used as is
        self.assertIsInstance(agent.tools["bash"], SmolagentsTool)
        self.assertIsInstance(agent.tools["bash"].tool, BashTool)
        self.assertIsInstance(agent.tools["python_interpreter"], PythonInterpreterTool)

        # Wrapped tools keep the txtai tool attributes and run it
        self.assertEqual(agent.tools["bash"].allowed, ["cat", "cut", "diff", "grep", "head", "ls", "tail"])
        self.assertEqual(agent.tools["bash"].inputs, BashTool().inputs)
        self.assertEqual(agent.tools["bash"]({"command": ["ls", tempfile.gettempdir()]}), BashTool()(["ls", tempfile.gettempdir()]))

        # Copies wrap the same txtai tool
        self.assertIs(copy.copy(agent.tools["bash"]).tool, agent.tools["bash"].tool)

        # Agent runs a tool call through the wrapped tool
        responses = iter(
            ['Action:\n{"name": "bash", "arguments": {"command": ["ls", "/"]}}', 'Action:\n{"name": "final_answer", "arguments": "Done"}']
        )
        agent = Agent(tools=["bash"], llm="hf-internal-testing/tiny-random-LlamaForCausalLM", max_steps=2)
        agent.process.model.llm = lambda *args, **kwargs: next(responses)

        with patch.object(BashTool, "forward", return_value="listing") as forward:
            self.assertEqual(agent("List files"), "Done")
            forward.assert_called_once_with(command=["ls", "/"])

    def testToolsWebSearch(self):
        """
        Test the web search tool parses search results
        """

        page = """
        <table>
          <tr><td><a class="result-link" href="#">txtai</a></td></tr>
          <tr><td class="result-snippet">All-in-one AI framework</td></tr>
          <tr><td><span class="link-text">github.com/neuml/txtai</span></td></tr>
          <tr><td><a class="result-link" href="#">txtai docs</a></td></tr>
          <tr><td><span class="link-text">neuml.github.io/txtai</span></td></tr>
        </table>
        """

        tool = ToolFactory.create({"tools": ["websearch"]})[0]
        self.assertIsInstance(tool, WebSearchTool)
        self.assertEqual(tool.name, "web_search")

        with patch.object(tool, "fetch", return_value=page.encode()) as fetch:
            self.assertEqual(
                tool("txtai framework"),
                "## Search Results\n\n[txtai](https://github.com/neuml/txtai)\nAll-in-one AI framework\n\n"
                "[txtai docs](https://neuml.github.io/txtai)\n",
            )
            fetch.assert_called_once_with("https://lite.duckduckgo.com/lite/?q=txtai+framework")

        with patch.object(tool, "fetch", return_value=b"<html></html>"), self.assertRaises(ValueError):
            tool("nothing")


class CustomProcess:
    """
    Custom agent process runner.
    """

    def __init__(self, model, tools, **kwargs):
        self.llm = model
        self.model = self
        self.tools = {tool.name: tool for tool in tools}
        self.config = kwargs
        self.maxlength = None

    def parameters(self, maxlength):
        """
        Sets LLM inference parameters.

        Args:
            maxlength: maximum sequence length
        """

        self.maxlength = maxlength

    # pylint: disable=W0613
    def run(self, text, stream=False, **kwargs):
        """
        Runs an agent loop.

        Args:
            text: instructions to run
            stream: stream response if True
            kwargs: additional keyword arguments

        Returns:
            result
        """

        return text.strip()
