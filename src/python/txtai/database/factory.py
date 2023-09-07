"""
Factory module
"""

from urllib.parse import urlparse

from ..util import Resolver

from .client import Client
from .duckdb import DuckDB
from .sqlite import SQLite


class DatabaseFactory:
    """
    Methods to create document databases.
    """

    @staticmethod
    def create(config):
        """
        Create a Database.

        Args:
            config: database configuration parameters

        Returns:
            Database
        """

        # Database instance
        database = None

        # Enables document database
        content = config.get("content")

        # Standardize content name
        if content is True:
            content = "sqlite"

        # Create document database instance
        if content == "duckdb":
            database = DuckDB(config)
        elif content == "sqlite":
            database = SQLite(config)
        elif content:
            # Check if content is a URL
            url = urlparse(content)
            if content == "client" or url.scheme:
                # Connect to database server URL
                database = Client(config)
            else:
                # Resolve custom database if content is not a URL
                database = DatabaseFactory.resolve(content, config)

        # Store config back
        config["content"] = content

        return database

    @staticmethod
    def resolve(backend, config):
        """
        Attempt to resolve a custom backend.

        Args:
            backend: backend class
            config: index configuration parameters

        Returns:
            Database
        """

        try:
            return Resolver()(backend)(config)
        except Exception as e:
            raise ImportError(f"Unable to resolve database backend: '{backend}'") from e
