"""
Placeholder module
"""


class Agent:
    """
    Agent placeholder stub for when agent dependencies aren't installed
    """

    # Set by txtai.agent.__init__ when the conditional import fails.
    # Contains the underlying ImportError message so users see the real
    # cause (e.g. mcpadapt/mcp incompatibility) instead of always being told
    # smolagents is missing. See issue #1161.
    _import_error = None

    def __init__(self, *args, **kwargs):
        """
        Raises an exception with the actual import failure reason.
        """

        msg = 'smolagents is not available - install "agent" extra to enable'
        if self._import_error:
            msg += f". Underlying import error: {self._import_error}"

        raise ImportError(msg)
