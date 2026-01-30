"""
Skill imports
"""

import yaml

from smolagents import Tool


class SkillTool(Tool):
    """
    Creates a SkillTool. A SkillTool loads a skill.md file.
    """

    # pylint: disable=W0231
    def __init__(self, path):
        """
        Creates a SkillTool.

        Args:
            path: skill file path
        """

        # Load skill.md
        metadata, content = self.load(path)

        # Tool parameters
        self.name = metadata["name"]
        self.description = metadata["description"]
        self.inputs = {"request": {"type": "string", "description": "The user requested action"}}
        self.output_type = "any"
        self.target = content

        # Validate parameters and initialize tool
        super().__init__()

    # pylint: disable=W0221
    def forward(self, request):
        """
        Searchs the skill markdown for the best answer.

        Args:
            request: user request

        Returns:
            result
        """

        return f"""Given the request {request}, find the best answer using the content below.

{self.target}
"""

    def load(self, path):
        """
        Loads a skill.md file.

        Args:
            path: path to skill.md

        Returns:
            metadata, content
        """

        # Read file content
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Stores frontmatter metadata, if any
        metadata = {}

        # Split by "---"" to separate frontmatter and markdown
        if content.startswith("---"):
            _, frontmatter, content = content.split("---", 2)
            metadata = yaml.safe_load(frontmatter)

        # Return parsed metadata and content
        return metadata, content
