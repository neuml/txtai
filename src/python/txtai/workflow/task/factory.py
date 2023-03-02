"""
Task factory module
"""

import functools

from ...util import Resolver


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
        return Resolver()(task)

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
                    config["action"] = [Partial.create(a, args[i]) for i, a in enumerate(action)]
                else:
                    # Accept keyword or positional arguments
                    config["action"] = lambda x: action(x, **args) if isinstance(args, dict) else action(x, *args)

        # Get Task instance
        return TaskFactory.get(task)(**config)


class Partial(functools.partial):
    """
    Modifies functools.partial to prepend arguments vs append.
    """

    @staticmethod
    def create(action, args):
        """
        Creates a new Partial function.

        Args:
            action: action to execute
            args: arguments

        Returns:
            Partial
        """

        return Partial(action, **args) if isinstance(args, dict) else Partial(action, *args) if args else Partial(action)

    def __call__(self, *args, **kwargs):
        # Update keyword arguments
        kw = self.keywords.copy()
        kw.update(kwargs)

        # Execute function with new arguments prepended to default arguments
        return self.func(*(args + self.args), **kw)
