"""
Factory module
"""

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
        if content == "sqlite":
            database = SQLite(config)

        # Store config back
        config["content"] = content

        return database
