"""
Placeholder module
"""


class Agent:
    """
    Agent placeholder for when agent dependencies cannot be imported
    """

    # Original dependency import failure
    ERROR = None

    def __init__(self, *args, **kwargs):
        """
        Raises an exception with installation guidance and the original cause.
        """

        raise ImportError('smolagents is not available - install "agent" extra to enable') from self.ERROR
