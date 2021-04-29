"""
Task factory module
"""


class TaskFactory:
    """
    Task factory. Creates new Task instances.
    """

    @staticmethod
    def get(task):
        """
        Gets a new instance of task class.

        Args:
            task: Task instance class

        Returns:
            Task class
        """

        # Local task if no package
        if "." not in task:
            # Get parent package
            task = ".".join(__name__.split(".")[:-1]) + "." + task.capitalize() + "Task"

        parts = task.split(".")
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)

        return m

    @staticmethod
    def create(config, task):
        """
        Creates a new Task instance.

        Args:
            config: Task configuration
            task: Task instance class

        Returns:
            Task
        """

        # Create lambda function if additional arguments present
        if "args" in config:
            args = config.pop("args")
            action = config["action"]
            config["action"] = lambda x: action(x, *args)

        # Get Task instance
        return TaskFactory.get(task)(**config)
