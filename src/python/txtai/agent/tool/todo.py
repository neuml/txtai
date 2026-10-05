"""
Todo imports
"""

from smolagents import Tool


class TodoWriteTool(Tool):
    """
    The TodoWriteTool plans for task execution.
    """

    # pylint: disable=W0231
    def __init__(self):
        """
        Creates a TodoWriteTool.
        """

        # Tool parameters
        self.name = "todowrite"
        self.description = (
            "Implementation of a todo write tool. Generates a structured task list to help organize complex tasks. "
            "Only use this tool for complex tasks with multiple steps. Skip for simple tasks that can be done easily."
        )
        self.inputs = {"plan": {"type": "string", "description": "The task plan"}}
        self.output_type = "any"

        # Validate parameters and initialize tool
        super().__init__()

    # pylint: disable=W0221
    def forward(self, plan):
        """
        Returns the plan.

        Returns:
            plan
        """

        return plan
