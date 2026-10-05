"""
Functions module
"""

from types import FunctionType, MethodType


class Functions:
    """
    Resolves function configuration to function references.
    """

    def __init__(self, embeddings):
        """
        Creates a new function resolver.

        Args:
            embeddings: embeddings instance
        """

        self.embeddings = embeddings

        # Handle to all reference objects
        self.references = None

    def __call__(self, config):
        """
        Resolves a list of functions to function references.

        Args:
            config: configuration

        Returns:
            list of function references
        """

        # Initialize stored references array
        self.references = []

        # Resolve callable functions
        functions = []
        for fn in config["functions"]:
            if isinstance(fn, dict):
                fn = fn.copy()
                fn["function"] = self.function(fn["function"])
            else:
                fn = self.function(fn)
            functions.append(fn)

        return functions

    def reset(self):
        """
        Clears all resolved references.
        """

        if self.references:
            for reference in self.references:
                reference.reset()

    def function(self, function):
        """
        Resolves function configuration. If function is a string, it's split on '.' and each part
        is separately resolved to an object, attribute or function. Each part is resolved upon the
        first invocation of the function. Otherwise, the input is returned.

        Args:
            function: function configuration

        Returns:
            function reference
        """

        if isinstance(function, str):
            parts = function.split(".")

            if hasattr(self.embeddings, parts[0]):
                m = Reference(self.embeddings, parts[0])
                self.references.append(m)
            else:
                module = ".".join(parts[:-1])
                m = __import__(module)

            for comp in parts[1:]:
                m = Reference(m, comp)
                self.references.append(m)

            return m

        return function


class Reference:
    """
    Stores a reference to an object attribute. This attribute is resolved by invoking the __call__ method.
    This allows for functions to be independent of the initialization order of an embeddings instance.
    """

    def __init__(self, obj, attribute):
        """
        Create a new reference.

        Args:
            obj: object handle
            attribute: attribute name
        """

        # Object handle and attribute
        self.obj = obj
        self.attribute = attribute

        # Keep a handle to the original inputs
        self.inputs = (obj, attribute)

        # True if the object and attribute have been resolved
        self.resolved = False

        # True if the attribute is a function
        self.function = None

    def __call__(self, *args):
        """
        Resolves an object attribute reference. If the attribute is a function, the function is executed.
        Otherwise, the object attribute value is returned.

        Args:
            args: list of function arguments to the object attribute, when attribute is a function

        Returns:
            object attribute function result or object attribute value
        """

        # Resolve nested function arguments, if necessary
        if not self.resolved:
            self.obj = self.obj() if isinstance(self.obj, Reference) else self.obj
            self.attribute = self.attribute() if isinstance(self.attribute, Reference) else self.attribute
            self.resolved = True

        # Lookup attribute
        attribute = getattr(self.obj, self.attribute)

        # Determine if attribute is a function
        if self.function is None:
            self.function = isinstance(attribute, (FunctionType, MethodType)) or (hasattr(attribute, "__call__") and args)

        # If attribute is a function, execute and return, otherwise return attribute
        return attribute(*args) if self.function else attribute

    def reset(self):
        """
        Clears resolved references.
        """

        self.obj, self.attribute = self.inputs
        self.resolved = False
