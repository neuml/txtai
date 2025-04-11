"""
Embeddings module
"""

from smolagents import Tool

from ...embeddings import Embeddings


class EmbeddingsTool(Tool):
    """
    Tool to execute an Embeddings search.
    """

    def __init__(self, config):
        """
        Creates a new EmbeddingsTool.

        Args:
            config: embeddings tool configuration
        """

        # Tool parameters
        self.name = config["name"]
        self.description = f"""{config['description']}. Results are returned as a list of dict elements.
Each result has keys 'id', 'text', 'score'."""

        # Input and output descriptions
        self.inputs = {"query": {"type": "string", "description": "The search query to perform."}}
        self.output_type = "any"

        # Load embeddings instance
        self.embeddings = self.load(config)

        # Validate parameters and initialize tool
        super().__init__()

    # pylint: disable=W0221
    def forward(self, query):
        """
        Runs a search.

        Args:
            query: input query

        Returns:
            search results
        """

        return self.embeddings.search(query, 5)

    def load(self, config):
        """
        Loads an embeddings instance from config.

        Args:
            config: embeddings tool configuration

        Returns:
            Embeddings
        """

        if "target" in config:
            return config["target"]

        embeddings = Embeddings()
        embeddings.load(**config)

        return embeddings
