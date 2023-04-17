"""
StreamTask module
"""

from .base import Task


class StreamTask(Task):
    """
    Task that calls a task action and yields results.
    """

    def register(self, batch=False):
        """
        Adds stream parameters to task.

        Args:
            batch: all elements are passed to a single action call if True, otherwise an action call is executed per element, defaults to False
        """

        # pylint: disable=W0201
        # All elements are passed to a single action call if True, otherwise an action call is executed per element, defaults to False
        self.batch = batch

    def __call__(self, elements, executor=None):
        for action in self.action:
            if self.batch:
                # Single batch call
                yield from action(elements)
            else:
                # Call action for each element
                for x in elements:
                    yield from action(x)
