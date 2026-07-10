"""
Keenable web search tool
"""

import json
import os

from urllib.request import Request, urlopen

from smolagents import Tool


class KeenableSearchTool(Tool):
    """
    Web search tool powered by the Keenable API. Keyless by default: it calls the public endpoint with no
    account or API key. Setting the KEENABLE_API_KEY environment variable only lifts the rate limit.
    """

    # pylint: disable=W0231
    def __init__(self):
        """
        Creates a KeenableSearchTool.
        """

        # Tool parameters - distinct name to avoid collision with smolagents WebSearchTool
        self.name = "keenable_search"
        self.description = (
            "Performs a web search using the Keenable API and returns relevant results. "
            "Use this tool to find current information on any topic. No API key is required."
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
        Runs a web search query using the Keenable API.

        Args:
            query: search query string

        Returns:
            formatted search results as a string
        """

        # Keyless by default; an optional key only lifts the rate limit
        key = os.environ.get("KEENABLE_API_KEY")
        url = "https://api.keenable.ai/v1/search" if key else "https://api.keenable.ai/v1/search/public"
        headers = {"Content-Type": "application/json", "X-Keenable-Title": "txtai"}
        if key:
            headers["X-API-Key"] = key

        data = json.dumps({"query": query, "mode": "pro"}).encode("utf-8")
        with urlopen(Request(url, data=data, headers=headers, method="POST"), timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))

        results = []
        for result in payload.get("results", [])[:5]:
            title = result.get("title", "")
            link = result.get("url", "")
            content = result.get("description", "")
            results.append(f"[{title}]({link})\n{content}")

        return "\n\n".join(results) if results else "No results found."
