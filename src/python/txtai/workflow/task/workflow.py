"""
WorkflowTask module
"""

from .base import Task


class WorkflowTask(Task):
    """
    Task that executes a separate Workflow
    """

    def process(self, action, inputs):
        return list(super().process(action, inputs))
