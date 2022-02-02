"""
Task factory module
"""

import functools


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

        # Attempt to load custom task
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
            if action:
                if isinstance(action, list):
                    config["action"] = [Partial(a, *args[i]) if args[i] else Partial(a) for i, a in enumerate(action)]
                else:
                    config["action"] = lambda x: action(x, *args)

        # Get Task instance
        return TaskFactory.get(task)(**config)


class Partial(functools.partial):
    """
    Modifies functools.partial to prepend arguments vs append.
    """

    def __call__(self, *args, **kwargs):
        # Update keyword arguments
        kw = self.keywords.copy()
        kw.update(kwargs)

        # Execute function with new arguments prepended to default arguments
        return self.func(*(args + self.args), **kw)
