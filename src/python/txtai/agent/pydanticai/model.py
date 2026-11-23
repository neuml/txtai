"""
Model module
"""

import json

from pydantic_ai.messages import (
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model

from ...pipeline import LLM


# pylint: disable=W0223
class PydanticModel(Model):
    """
    Pydantic AI model backed by a LLM pipeline.
    """

    def __init__(self, path=None, method=None, **kwargs):
        """
        Creates a new LLM model.

        Args:
            path: model path or instance
            method: llm model framework, infers from path if not provided
            kwargs: model keyword arguments
        """

        super().__init__()

        self.llm = path if isinstance(path, LLM) else LLM(path, method, **kwargs)
        self.maxlength = 8192

    @property
    def model_name(self):
        return self.llm.generator.path if isinstance(self.llm, LLM) else "txtai"

    @property
    def system(self):
        return "txtai"

    async def request(self, messages, model_settings, model_request_parameters):
        """
        Runs LLM inference. This method signature must match the Pydantic AI specification.

        Args:
            messages: list of Pydantic AI messages
            model_settings: model settings
            model_request_parameters: request parameters, includes the list of tools

        Returns:
            ModelResponse
        """

        _, parameters = self.prepare_request(model_settings, model_request_parameters)

        # Build chat messages and run LLM inference
        response = self.llm(self.messages(messages, parameters), maxlength=self.maxlength, stop=["Observation:"])

        return ModelResponse(parts=[self.parse(response)], model_name=self.model_name)

    def parameters(self, maxlength):
        """
        Set LLM inference parameters.

        Args:
            maxlength: maximum sequence length
        """

        self.maxlength = maxlength

    def messages(self, messages, parameters):
        """
        Converts Pydantic AI messages to a list of chat messages.

        Args:
            messages: list of Pydantic AI messages
            parameters: request parameters

        Returns:
            list of chat messages
        """

        chat = []

        # Add tool instructions
        tools = parameters.function_tools + parameters.output_tools
        if tools:
            chat.append({"role": "system", "content": self.instructions(tools, parameters.allow_text_output)})

        for message in messages:
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    chat.append({"role": "system", "content": part.content})
                elif isinstance(part, UserPromptPart):
                    content = part.content if isinstance(part.content, str) else "\n".join(str(x) for x in part.content)
                    chat.append({"role": "user", "content": content})
                elif isinstance(part, ToolReturnPart):
                    chat.append({"role": "user", "content": f"Observation: {self.observation(part)}"})
                elif isinstance(part, RetryPromptPart):
                    chat.append({"role": "user", "content": part.model_response()})
                elif isinstance(part, TextPart):
                    chat.append({"role": "assistant", "content": part.content})
                elif isinstance(part, ToolCallPart):
                    action = json.dumps({"name": part.tool_name, "arguments": part.args_as_dict()})
                    chat.append({"role": "assistant", "content": f"Action:\n{action}"})

        # Merge consecutive messages with the same role, some chat templates require alternating roles
        merged = []
        for message in chat:
            if merged and merged[-1]["role"] == message["role"]:
                merged[-1]["content"] += f"\n\n{message['content']}"
            else:
                merged.append(dict(message))

        return merged

    def instructions(self, tools, text):
        """
        Builds tool calling instructions.

        Args:
            tools: list of tool definitions
            text: True if a plain text final answer is allowed

        Returns:
            instructions
        """

        answer = (
            "When you have the final answer, respond with the answer as plain text without an Action."
            if text
            else "When you have the final answer, call one of the final result tools."
        )

        lines = [
            "You solve tasks step by step using the tools below. To call a tool, respond with only the following format:",
            "",
            "Action:",
            '{"name": "tool name", "arguments": {"argument name": "value"}}',
            "",
            f"The tool result is returned as an Observation. {answer}",
            "",
            "Tools:",
        ]

        for tool in tools:
            lines.append(f"- {tool.name}: {tool.description or ''}")
            lines.append(f"  arguments: {json.dumps(tool.parameters_json_schema.get('properties', {}))}")

        return "\n".join(lines)

    def observation(self, part):
        """
        Formats a tool result as text.

        Args:
            part: ToolReturnPart

        Returns:
            tool result text
        """

        # pylint: disable=W0718
        try:
            return part.model_response_str()
        except Exception:
            return str(part.content)

    def parse(self, response):
        """
        Parses a LLM response into a text answer or a tool call.

        Args:
            response: LLM response text

        Returns:
            TextPart or ToolCallPart
        """

        # Remove anything generated after a stop sequence
        response = response.split("Observation:")[0]

        index = response.find("Action:")
        if index == -1:
            return TextPart(response.strip())

        action = response[index + len("Action:") :].strip()
        try:
            action, _ = json.JSONDecoder().raw_decode(action)
        except json.JSONDecodeError:
            # Invalid tool calls are sent back to the model to retry
            return ToolCallPart(tool_name="", args=action)

        if not isinstance(action, dict):
            return ToolCallPart(tool_name="", args=json.dumps(action))

        name, arguments = action.get("name", ""), action.get("arguments", {})

        # Support the final_answer action format
        if name == "final_answer":
            answer = arguments.get("answer", next(iter(arguments.values()), "")) if isinstance(arguments, dict) else arguments
            return TextPart(str(answer))

        return ToolCallPart(tool_name=name, args=arguments if isinstance(arguments, dict) else str(arguments))
