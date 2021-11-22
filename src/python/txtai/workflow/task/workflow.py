"""
WorkflowTask module
"""

from .base import Task


class WorkflowTask(Task):
    """
    Task that executes a separate Workflow
    """

    def __call__(self, elements):
        return [list(result) for result in super().__call__(elements)]
