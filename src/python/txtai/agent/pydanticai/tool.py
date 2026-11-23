"""
Tool module
"""

from pydantic_ai import Tool


class PydanticTool:
    """
    Converts txtai agent tools to Pydantic AI tools.
    """

    # Input types that map directly to JSON schema types
    TYPES = ["array", "boolean", "integer", "null", "number", "object", "string"]

    @staticmethod
    def create(tool):
        """
        Creates a Pydantic AI tool from a txtai agent tool.

        Args:
            tool: txtai agent tool

        Returns:
            Pydantic AI tool
        """

        # pylint: disable=W0108
        return Tool.from_schema(
            lambda **kwargs: tool(**kwargs),
            name=tool.name,
            description=tool.description,
            json_schema=PydanticTool.schema(tool.inputs),
        )

    @staticmethod
    def schema(inputs):
        """
        Builds a JSON schema from tool inputs.

        Args:
            inputs: tool inputs dictionary

        Returns:
            JSON schema
        """

        properties, required = {}, []
        for name, config in inputs.items():
            # Unknown types (i.e. any) accept any value
            parameter = {"type": config["type"]} if config.get("type") in PydanticTool.TYPES else {}
            if config.get("description"):
                parameter["description"] = config["description"]

            properties[name] = parameter

            # Nullable inputs are optional
            if not config.get("nullable"):
                required.append(name)

        return {"type": "object", "properties": properties, "required": required}
