"""
Agent module
"""

import os

from collections import deque

from jinja2 import Template

from .factory import ProcessFactory


class Agent:
    """
    An agent automatically creates workflows to answer multi-faceted user requests. Agents iteratively prompt and/or interface with tools to
    step through a process and ultimately come to an answer for a request.

    Agents excel at complex tasks where multiple tools and/or methods are required. They incorporate a level of randomness similar to different
    people working on the same task. When the request is simple and/or there is a rule-based process, other methods such as RAG and Workflows
    should be explored.
    """

    def __init__(self, template=None, memory=None, **kwargs):
        """
        Creates a new Agent.

        Args:
            template: optional prompt jinja template, must include {{ text }} and {{ memory }} placeholders
            memory: number of prior outputs to keep as "memory", defaults to None for no memory
            kwargs: arguments to pass to the underlying Agent backend and LLM pipeline instance
        """

        # Ensure backwards compatibility
        if "max_iterations" in kwargs:
            kwargs["max_steps"] = kwargs.pop("max_iterations")

        # Custom instructions
        if "instructions" in kwargs:
            kwargs["instructions"] = self.instructions(kwargs)

        # Create agent process runner
        self.process = ProcessFactory.create(kwargs)

        # Tools dictionary
        self.tools = self.process.tools

        # Agent memory
        self.memory = {}
        self.window = memory
        self.template = template

    def __call__(self, text, maxlength=8192, stream=False, session=None, reset=False, **kwargs):
        """
        Runs an agent loop.

        Args:
            text: instructions to run
            maxlength: maximum sequence length
            stream: stream response if True, defaults to False
            session: session id for stored memory, defaults to None which shares all memory
            reset: clears previously stored memory if True, defaults to False
            kwargs: additional keyword arguments

        Returns:
            result
        """

        # Process parameters
        self.process.model.parameters(maxlength)

        # Create memory, if necessary
        if self.window and session not in self.memory:
            self.memory[session] = deque(maxlen=self.window)

        # Clear memory, if reset flag set
        if reset and session in self.memory:
            self.memory[session].clear()

        # Run agent loop
        output = self.process.run(self.prompt(text, session), stream=stream, **kwargs)

        # Add output to memory, if necessary
        if session in self.memory:
            self.memory[session].append((text, output))

        return output

    def prompt(self, text, session):
        """
        Generates the full agent prompt using the input instructions. Adds in agent memory
        if available.

        Args:
            text: instructions to run
            session: session id for stored memory

        Returns:
            formatted instructions
        """

        template = (
            self.template
            if self.template
            else """{{ text }}
{% if memory %}
Use the following conversation history to help answer the question above.

{{ memory }}

If the history is irrelevant, forget it and use other tools to answer the question.
{% endif %}
"""
        )

        # pylint: disable=E1133
        memory = []
        if self.memory.get(session):
            # Add memory context
            for request, output in self.memory[session]:
                memory.append(f"User: {request}\nAssistant: {output}")

            memory = "\n\n".join(memory)

        # Command template with memory
        return Template(template).render(text=text, memory=memory)

    def instructions(self, config):
        """
        Reads and formats custom instructions. Supports agents.md files.

        Args:
            config: agent configuration

        Returns:
            agent instructions, if any
        """

        # Check if this is a file path (i.e. agents.md)
        path = config.pop("instructions", None)
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                # Read entire file
                return f"\nBelow is a set of general instructions to follow:\n\n{f.read()}"

        return path
