"""
AutoId module
"""

import inspect
import uuid


class AutoId:
    """
    Generates unique ids.
    """

    def __init__(self, method=None):
        """
        Creates a unique id generator.

        Args:
            method: generation method - supports int sequence (default) or UUID function
        """

        # Initialize variables
        self.method, self.function, self.value = None, None, None

        # Set id generation method
        if not method or isinstance(method, int):
            # Incrementing sequence (default)
            self.method = self.sequence
            self.value = method if method else 0
        else:
            # UUID generation function
            self.method = self.uuid
            self.function = getattr(uuid, method)

        # Check if signature takes a namespace argument (deterministic)
        args = inspect.getfullargspec(self.function).args if self.function else []
        self.deterministic = "namespace" in args

    def __call__(self, data=None):
        """
        Generates a unique id.

        Args:
            data: optional data to use for deterministic algorithms (i.e. uuid3, uuid5)

        Returns:
            unique id
        """

        return self.method(data)

    # pylint: disable=W0613
    def sequence(self, data):
        """
        Gets and increments sequence.

        Args:
            data: not used

        Returns:
            current sequence value
        """

        # Get and increment sequence
        value = self.value
        self.value += 1

        return value

    def uuid(self, data):
        """
        Generates a UUID and return as a string.

        Args:
            data: used with determistic algorithms (uuid3, uuid5)

        Returns:
            UUID string
        """

        uid = self.function(uuid.NAMESPACE_DNS, str(data)) if self.deterministic else self.function()
        return str(uid)

    def current(self):
        """
        Get the current sequence value. Only applicable for sequence ids, will be None for UUID methods.

        Returns:
            current sequence value
        """

        return self.value
