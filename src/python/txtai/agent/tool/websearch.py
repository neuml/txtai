"""
Web search imports
"""

from urllib.parse import quote_plus

# Conditional import
try:
    from bs4 import BeautifulSoup

    BS4 = True
except ImportError:
    BS4 = False

from .base import Tool

from ...pipeline import URLRetrieve


class WebSearchTool(Tool):
    """
    The WebSearchTool runs a DuckDuckGo web search and returns the top results.
    """

    # pylint: disable=W0231
    def __init__(self, maxresults=10):
        """
        Creates a WebSearchTool.

        Args:
            maxresults: maximum number of results to return
        """

        if not BS4:
            raise ImportError('WebSearchTool is not available - install "agent" extra to enable')

        # Tool parameters
        self.name = "web_search"
        self.description = (
            "Performs a web search for a query and returns a string of the top search results formatted as markdown "
            "with titles, links, and descriptions."
        )
        self.inputs = {"query": {"type": "string", "description": "The search query to perform."}}
        self.output = "string"

        # URL retrieval pipeline
        self.fetch = URLRetrieve(headers={"User-Agent": "Mozilla/5.0"}, safeopen=True)
        self.maxresults = maxresults

        # Validate parameters and initialize tool
        super().__init__()

    # pylint: disable=W0221
    def forward(self, query):
        """
        Runs a web search.

        Args:
            query: search query

        Returns:
            search results as markdown
        """

        results = self.parse(self.fetch(f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"))
        if not results:
            raise ValueError("No results found! Try a less restrictive/shorter query.")

        return "## Search Results\n\n" + "\n\n".join(
            f"[{result['result-link']}](https://{result['link-text']})\n{result.get('result-snippet', '')}" for result in results
        )

    def parse(self, html):
        """
        Parses a DuckDuckGo Lite results page.

        Args:
            html: results page

        Returns:
            list of {"result-link", "result-snippet", "link-text"}
        """

        soup = BeautifulSoup(html, features="html.parser")

        # Each result is a title, then a snippet (optional), then the link text
        results, current = [], {}
        for element in soup.find_all(class_=["result-link", "result-snippet", "link-text"]):
            name = element.get("class")[0]
            current[name] = element.text.strip()
            if name == "link-text":
                if "result-link" in current:
                    results.append(current)
                current = {}

        return results[: self.maxresults]
