"""
ConsoleTask module
"""

import json

from .base import Task


class ConsoleTask(Task):
    """
    Task that prints task elements to the console.
    """

    def __call__(self, elements, executor=None):
        # Run task
        outputs = super().__call__(elements, executor)

        # Print inputs and outputs to console
        print("Inputs:", json.dumps(elements, indent=2))
        print("Outputs:", json.dumps(outputs, indent=2))

        # Return results
        return outputs
