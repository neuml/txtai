"""
Agent module
"""

from .factory import ProcessFactory


class Agent:
    """
    An agent automatically creates workflows to answer multi-faceted user requests. Agents iteratively prompt and/or interface with tools to
    step through a process and ultimately come to an answer for a request.

    Agents excel at complex tasks where multiple tools and/or methods are required. They incorporate a level of randomness similar to different
    people working on the same task. When the request is simple and/or there is a rule-based process, other methods such as RAG and Workflows
    should be explored.
    """

    def __init__(self, **kwargs):
        """
        Creates a new Agent.

        Args:
            kwargs: arguments to pass to the underlying Agent backend and LLM pipeline instance
        """

        # Ensure backwards compatibility
        if "max_iterations" in kwargs:
            kwargs["max_steps"] = kwargs.pop("max_iterations")

        # Create agent process runner
        self.process = ProcessFactory.create(kwargs)

        # Tools dictionary
        self.tools = self.process.tools

    def __call__(self, text, maxlength=8192, stream=False, **kwargs):
        """
        Runs an agent loop.

        Args:
            text: instructions to run
            maxlength: maximum sequence length
            stream: stream response if True, defaults to False
            kwargs: additional keyword arguments

        Returns:
            result
        """

        # Process parameters
        self.process.model.parameters(maxlength)

        # Run agent loop
        return self.process.run(text, stream=stream, **kwargs)
