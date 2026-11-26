"""
Base imports
"""


class Tool:
    """
    Base class for txtai agent tools. A tool has a name, description, inputs and an output type. Calling a tool runs its forward method.

    This class doesn't depend on an agent framework. Each agent backend converts tools to its own format.
    """

    # Tool parameters
    name: str
    description: str
    inputs: dict
    output_type: str

    def __init__(self):
        """
        Validates tool parameters. Subclasses set the `name`, `description`, `inputs` and `output_type` attributes before calling this method.
        """

        for name, types in [("name", str), ("description", str), ("inputs", dict), ("output_type", str)]:
            value = getattr(self, name, None)
            if not isinstance(value, types):
                raise TypeError(f"Attribute {name} should have type {types.__name__}, got {type(value).__name__} instead.")

    def __call__(self, *args, **kwargs):
        """
        Runs the tool.

        Args:
            args: positional arguments, a single dictionary of named inputs is also accepted
            kwargs: keyword arguments

        Returns:
            tool output
        """

        # Arguments can be passed as a single dictionary of named inputs
        if len(args) == 1 and not kwargs and isinstance(args[0], dict) and all(key in self.inputs for key in args[0]):
            args, kwargs = (), args[0]

        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Runs the tool logic. Subclasses must implement this method.

        Args:
            args: positional arguments
            kwargs: keyword arguments

        Returns:
            tool output
        """

        raise NotImplementedError
