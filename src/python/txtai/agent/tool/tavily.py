"""
Tavily module
"""

import os

from smolagents import Tool


class TavilySearchTool(Tool):
    """
    Web search tool using the Tavily API.
    """

    name = "tavily_search"
    description = "Performs a web search using the Tavily API and returns the top results."
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        try:
            from tavily import TavilyClient  # pylint: disable=C0415
        except ImportError as e:
            raise ImportError("tavily-python is required to use the tavily tool. Install it with: pip install tavily-python") from e

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError("TAVILY_API_KEY environment variable is required to use the tavily tool.")

        self.client = TavilyClient(api_key=api_key)

    def forward(self, query: str) -> str:
        """
        Runs a Tavily web search and returns formatted results.

        Args:
            query: search query

        Returns:
            formatted search results as a string
        """

        response = self.client.search(query=query, max_results=5, search_depth="basic")
        results = []
        for r in response.get("results", []):
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            results.append(f"[{title}]({url})\n{content}")

        return "\n\n".join(results) if results else "No results found."
