"""
Question imports
"""

from .base import Tool


class QuestionTool(Tool):
    """
    The QuestionTool asks the user a question and returns the answer.
    """

    # pylint: disable=W0231
    def __init__(self):
        """
        Creates a QuestionTool.
        """

        # Tool parameters
        self.name = "user_input"
        self.description = "Asks for user's input on a specific question"
        self.inputs = {"question": {"type": "string", "description": "The question to ask the user"}}
        self.output_type = "string"

        # Validate parameters and initialize tool
        super().__init__()

    # pylint: disable=W0221
    def forward(self, question):
        """
        Asks the user a question.

        Args:
            question: question to ask

        Returns:
            answer
        """

        return input(f"{question} => Type your answer here:")
