"""
Resolver module
"""


class Resolver:
    """
    Resolves a Python class path
    """

    def __call__(self, path):
        """
        Class instance to resolve.

        Args:
            path: path to class

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

        # Return class instance
        return m
