"""
Process module
"""

# Conditional import
try:
    from pydantic_ai import Agent
    from pydantic_ai.usage import UsageLimits

    from .model import PydanticModel
    from .tool import PydanticTool

    PYDANTIC_AI = True
except ImportError:
    PYDANTIC_AI = False


class PydanticProcess:
    """
    Agent process runner backed by Pydantic AI.
    """

    def __init__(self, model, tools, instructions=None, max_steps=20, method=None, **kwargs):
        """
        Creates a new PydanticProcess.

        Args:
            model: model path, model configuration dictionary, LLM pipeline or PydanticModel
            tools: list of txtai agent tools
            instructions: optional custom instructions
            max_steps: maximum number of LLM calls per run
            method: agent method, only the default tool calling method is supported
            kwargs: additional keyword arguments to pass to the Pydantic AI Agent
        """

        if not PYDANTIC_AI:
            raise ImportError('Pydantic AI is not available - install "agent" extra to enable')

        if method and method != "tool":
            raise ValueError(f'Agent method "{method}" is not supported by the pydantic backend')

        # Create model backed by LLM pipeline
        if not isinstance(model, PydanticModel):
            model = PydanticModel(**model) if isinstance(model, dict) else PydanticModel(model)

        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.maxsteps = max_steps

        self.agent = Agent(
            self.model,
            system_prompt=instructions if instructions else (),
            tools=[PydanticTool.create(tool) for tool in tools],
            **kwargs,
        )

    def run(self, text, stream=False, **kwargs):
        """
        Runs an agent loop.

        Args:
            text: instructions to run
            stream: when True, returns a generator that yields the final answer
            kwargs: additional keyword arguments to pass to the Pydantic AI Agent run

        Returns:
            result
        """

        output = self.agent.run_sync(text, usage_limits=UsageLimits(request_limit=self.maxsteps), **kwargs).output
        return self.generate(output) if stream else output

    def generate(self, output):
        """
        Yields output as a stream.

        Args:
            output: agent output

        Returns:
            generator
        """

        yield output
