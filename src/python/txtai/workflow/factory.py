"""
Workflow factory module
"""

from .base import Workflow
from .task import TaskFactory


class WorkflowFactory:
    """
    Workflow factory. Creates new Workflow instances.
    """

    @staticmethod
    def create(config, name):
        """
        Creates a new Workflow instance.

        Args:
            config: Workflow configuration
            name: Workflow name

        Returns:
            Workflow
        """

        # Resolve workflow tasks
        tasks = []
        for tconfig in config["tasks"]:
            task = tconfig.pop("task") if "task" in tconfig else ""
            tasks.append(TaskFactory.create(tconfig, task))

        config["tasks"] = tasks

        if "stream" in config:
            sconfig = config["stream"]
            task = sconfig.pop("task") if "task" in sconfig else "stream"

            config["stream"] = TaskFactory.create(sconfig, task)

        # Create workflow
        return Workflow(**config, name=name)
