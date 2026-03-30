"""
Tavily web search tool
"""

import os

from smolagents import Tool


class TavilySearchTool(Tool):
    """
    Web search tool powered by the Tavily API. Requires the TAVILY_API_KEY environment variable to be set.
    """

    # pylint: disable=W0231
    def __init__(self):
        """
        Creates a TavilySearchTool.
        """

        # Tool parameters
        self.name = "web_search"
        self.description = (
            "Performs a web search using the Tavily API and returns relevant results. "
            "Use this tool to find current information on any topic."
        )
        self.inputs = {
            "query": {"type": "string", "description": "The search query to look up on the web."}
        }
        self.output_type = "string"

        # Validate parameters and initialize tool
        super().__init__()

    # pylint: disable=W0221
    def forward(self, query):
        """
        Runs a web search query using the Tavily API.

        Args:
            query: search query string

        Returns:
            formatted search results as a string
        """

        # Import here to make tavily-python an optional dependency
        from tavily import TavilyClient  # pylint: disable=C0415

        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        response = client.search(query=query, max_results=5, search_depth="basic")

        results = []
        for result in response.get("results", []):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")
            results.append(f"[{title}]({url})\n{content}")

        return "\n\n".join(results) if results else "No results found."
