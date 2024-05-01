"""
Utils module
"""

import requests


class Utils:
    """
    Utility constants and methods
    """

    PATH = "/tmp/txtai"


def qdrant_server_running() -> bool:
    """Check if Qdrant server is running."""

    try:
        response = requests.get("http://localhost:6333", timeout=10.0)
        response_json = response.json()
        return response_json.get("title") == "qdrant - vector search engine"
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False
