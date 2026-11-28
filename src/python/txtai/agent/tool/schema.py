"""
Schema module
"""

import inspect
import json
import re
import types

from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

# Type of None
NONE = type(None)


class TypeHintError(TypeError):
    """
    Raised when a type hint can't be converted to a JSON schema.
    """


class DocstringError(ValueError):
    """
    Raised when a docstring is missing or doesn't describe every argument.
    """


class ToolSchema:
    """
    Builds JSON schemas for tool functions from type hints and Google format docstrings.
    """

    # Docstring sections
    DESCRIPTION = re.compile(r"^(.*?)[\n\s]*(Args:|Returns:|Raises:|\Z)", re.DOTALL)
    ARGS = re.compile(r"\n\s*Args:\n(.*?)[\n\s]*(Returns:|Raises:|\Z)", re.DOTALL)
    RETURNS = re.compile(r"\n\s*Returns:\n\s*(.*?)[\n\s]*(Raises:|\Z)", re.DOTALL)

    # Splits the Args section into arguments. Each argument is a "name:" or "name (type):" line followed by
    # its description, which ends at the next "name:" line that isn't indented deeper.
    ARG = re.compile(
        r"(?:^|\n)([ \t]*)(\w+)(?:[ \t]*\([^()\n]*\))?:\s*(.*?)\s*(?=\n(?!\1[ \t])[ \t]*\w+(?:[ \t]*\([^()\n]*\))?:|\Z)",
        re.DOTALL,
    )

    # JSON scalar types
    SCALARS = {int: "integer", float: "number", str: "string", bool: "boolean", NONE: "null"}

    # Bare types
    TYPES = {**SCALARS, list: "array", tuple: "array", dict: "object"}

    # Types matched by name, so their packages aren't imported
    NAMED = {("PIL.Image", "Image"): "image", ("torch", "Tensor"): "audio"}

    @staticmethod
    def create(function):
        """
        Builds a JSON schema for function. The function needs a docstring that describes each argument and
        a type hint for each argument.

        Args:
            function: input function

        Returns:
            {"name", "description", "parameters", "return"} - return is only set when the return type hint converts
        """

        name = getattr(function, "__name__", "operation")

        doc = inspect.getdoc(function)
        if not doc:
            raise DocstringError(f"Cannot generate JSON schema for {name} because it has no docstring!")

        description, arguments, returns = ToolSchema.docstring(doc.strip())

        parameters = ToolSchema.parameters(function)
        output = parameters["properties"].pop("return", None)
        if output is not None and returns is not None:
            output["description"] = returns

        for argument, schema in parameters["properties"].items():
            if argument not in arguments:
                raise DocstringError(f"Cannot generate JSON schema for {name} because the docstring has no description for the argument '{argument}'")

            text = arguments[argument]
            choices = re.search(r"\(choices:\s*(.*?)\)\s*$", text, flags=re.IGNORECASE)
            if choices:
                schema["enum"] = [x.strip() if isinstance(x, str) else x for x in json.loads(choices.group(1))]
                text = text[: choices.start()].strip()

            schema["description"] = text

        schema = {"name": name, "description": description, "parameters": parameters}
        if output is not None:
            schema["return"] = output

        return schema

    @staticmethod
    def docstring(doc):
        """
        Parses a Google format docstring.

        Args:
            doc: docstring

        Returns:
            (description, {argument: description}, returns description)
        """

        description = ToolSchema.DESCRIPTION.search(doc)
        arguments = ToolSchema.ARGS.search(doc)
        returns = ToolSchema.RETURNS.search(doc)

        description = description.group(1).strip() if description else None
        returns = returns.group(1).strip() if returns else None

        parsed = {}
        if arguments:
            arguments = "\n".join(line for line in arguments.group(1).split("\n") if line.strip())
            for _, name, text in ToolSchema.ARG.findall(arguments):
                parsed[name] = re.sub(r"\s*\n+\s*", " ", text.strip())

        return description, parsed, returns

    @staticmethod
    def parameters(function):
        """
        Builds a JSON schema for the arguments and return type of function.

        Args:
            function: input function

        Returns:
            {"type": "object", "properties", "required"} - the return type is in properties when it converts
        """

        hints = get_type_hints(function)
        signature = inspect.signature(function)
        name = getattr(function, "__name__", "operation")

        # An untyped self or cls first argument is the receiver of a method
        first = next(iter(signature.parameters), None)
        receiver = first if first in ("self", "cls") and signature.parameters[first].annotation == inspect.Parameter.empty else None

        required = []
        for pname, parameter in signature.parameters.items():
            if pname != receiver:
                if parameter.annotation == inspect.Parameter.empty:
                    raise TypeHintError(f"Argument {pname} is missing a type hint in function {name}")
                if parameter.default == inspect.Parameter.empty:
                    required.append(pname)

        properties = {}
        for pname, hint in hints.items():
            if pname == "return":
                # The return value isn't generated by the model, leave out return types that don't convert
                try:
                    properties[pname] = ToolSchema.hint(hint)
                except TypeHintError:
                    pass
            elif pname != receiver:
                properties[pname] = ToolSchema.hint(hint)

        schema = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required

        return schema

    @staticmethod
    def hint(hint):
        """
        Converts a type hint to a JSON schema.

        Args:
            hint: type hint

        Returns:
            JSON schema
        """

        origin, args = get_origin(hint), get_args(hint)

        if origin is None:
            return ToolSchema.bare(hint)

        if origin is Union or origin is types.UnionType:
            return ToolSchema.union(args)

        if origin is Literal and args:
            if any(x.__class__ not in ToolSchema.SCALARS for x in args):
                raise TypeHintError("Only the valid python literals can be listed in typing.Literal.")
            return ToolSchema.enum(list(args))

        if origin in (list, tuple, dict):
            return ToolSchema.container(hint, origin, args)

        raise TypeHintError(f"Couldn't parse the type hint {hint!r}, likely due to a custom class or object")

    @staticmethod
    def union(args):
        """
        Converts the type arguments of a Union to a JSON schema.

        Args:
            args: type arguments

        Returns:
            JSON schema
        """

        # None makes the type nullable
        subtypes = [ToolSchema.hint(x) for x in args if x is not NONE]
        if len(subtypes) == 1:
            schema = subtypes[0]
        elif all(x.keys() == {"type"} and isinstance(x["type"], str) for x in subtypes):
            # Unions of basic types are a list of types
            names = sorted({x["type"] for x in subtypes})
            schema = {"type": names[0] if len(names) == 1 else names}
        else:
            schema = {"anyOf": subtypes}

        if NONE in args:
            schema["nullable"] = True

        return schema

    @staticmethod
    def container(hint, origin, args):
        """
        Converts a list, tuple or dict type hint to a JSON schema.

        Args:
            hint: type hint
            origin: list, tuple or dict
            args: type arguments

        Returns:
            JSON schema
        """

        if origin is list:
            return {"type": "array", "items": ToolSchema.hint(args[0])} if args else {"type": "array"}

        if origin is tuple:
            if not args:
                return {"type": "array"}
            if len(args) == 1 or ... in args:
                raise TypeHintError(f"The type hint {str(hint).replace('typing.', '')} isn't supported, use a list[] type instead")
            return {"type": "array", "prefixItems": [ToolSchema.hint(x) for x in args]}

        # The keys of a JSON object are strings, the type of its values can be set
        schema = {"type": "object"}
        if len(args) == 2:
            schema["additionalProperties"] = ToolSchema.hint(args[1])
        return schema

    @staticmethod
    def bare(hint):
        """
        Converts a type hint without type arguments to a JSON schema.

        Args:
            hint: type hint

        Returns:
            JSON schema
        """

        # Enum classes are an enum of their member values
        if isinstance(hint, type) and issubclass(hint, Enum):
            values = [member.value for member in hint]
            if not values:
                raise TypeHintError(f"The Enum {hint.__name__} has no members, so no value could ever match it.")
            if any(value.__class__ not in ToolSchema.SCALARS for value in values):
                raise TypeHintError(f"The Enum {hint.__name__} has member values that cannot be written as JSON scalars.")
            return ToolSchema.enum(values)

        if hint is Any:
            return {}

        if isinstance(hint, type):
            if hint in ToolSchema.TYPES:
                return {"type": ToolSchema.TYPES[hint]}

            named = ToolSchema.NAMED.get((hint.__module__, hint.__name__))
            if named:
                return {"type": named}

        raise TypeHintError(
            f"Couldn't convert the type hint {hint!r} to a JSON schema, likely because it is a custom class or object. "
            "Supported type hints are basic types like int, float, str and bool, Enum classes, typing.Literal, "
            "and list, tuple, dict and Union types containing those."
        )

    @staticmethod
    def enum(values):
        """
        Builds a JSON schema enum.

        Args:
            values: list of JSON scalar values

        Returns:
            JSON schema
        """

        # Each JSON type in the values, once, in the order they first appear
        names = list(dict.fromkeys(ToolSchema.SCALARS[value.__class__] for value in values))
        return {"type": names[0] if len(names) == 1 else names, "enum": values}
