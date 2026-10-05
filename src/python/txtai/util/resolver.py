"""
Resolver module
"""


class Resolver:
    """
    Resolves a Python class path
    """

    def __call__(self, path, base=None):
        """
        Class instance to resolve.

        Args:
            path: path to class
            base: optional required base class

        Returns:
            class instance
        """

        # Split into path components
        parts = path.split(".")

        # Resolve each path component
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)

        # Validate base class requirement, if necessary
        if base and (not isinstance(m, type) or not issubclass(m, base)):
            raise ImportError(f"{path} is not a subclass of {base.__name__}")

        # Return class instance
        return m
