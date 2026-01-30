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
        self.memory = deque(maxlen=memory) if memory else None
        self.template = template

    def __call__(self, text, maxlength=8192, stream=False, reset=False, **kwargs):
        """
        Runs an agent loop.

        Args:
            text: instructions to run
            maxlength: maximum sequence length
            stream: stream response if True, defaults to False
            reset: clears previously stored memory if True, defaults to False
            kwargs: additional keyword arguments

        Returns:
            result
        """

        # Process parameters
        self.process.model.parameters(maxlength)

        # Clear memory, if reset flag set
        if reset and self.memory:
            self.memory.clear()

        # Run agent loop
        output = self.process.run(self.prompt(text), stream=stream, **kwargs)

        # Add output to memory, if necessary
        if self.memory is not None:
            self.memory.append((text, output))

        return output

    def prompt(self, text):
        """
        Generates the full agent prompt using the input instructions. Adds in agent memory
        if available.

        Args:
            text: instructions to run

        Returns:
            formatted instructions
        """

        template = (
            self.template
            if self.template
            else """{{ text }}
{% if memory %}
Research carefully before answering. This is what you've currently learned. Only review if
it's relevant to the current topic above.

{{ memory }}

{% endif %}
"""
        )

        # pylint: disable=E1133
        memory = []
        if self.memory:
            # Add memory context
            for request, output in self.memory:
                memory.append(f"Input: {request}\nOutput: {output}")

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
                return f"Below is a set of general instructions to follow:\n\n{f.read()}"

        return path
