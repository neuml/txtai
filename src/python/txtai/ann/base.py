"""
ANN (Approximate Nearest Neighbor) module
"""

import datetime
import platform

from ..version import __version__


class ANN:
    """
    Base class for ANN instances. This class builds vector indexes to support similarity search.
    The built-in ANN backends store ids and vectors. Content storage is supported via database instances.
    """

    def __init__(self, config):
        """
        Creates a new ANN.

        Args:
            config: index configuration parameters
        """

        # ANN index
        self.backend = None

        # ANN configuration
        self.config = config

    def load(self, path):
        """
        Loads an ANN at path.

        Args:
            path: path to load ann index
        """

        raise NotImplementedError

    def index(self, embeddings):
        """
        Builds an ANN index.

        Args:
            embeddings: embeddings array
        """

        raise NotImplementedError

    def append(self, embeddings):
        """
        Append elements to an existing index.

        Args:
            embeddings: embeddings array
        """

        raise NotImplementedError

    def delete(self, ids):
        """
        Deletes elements from existing index.

        Args:
            ids: ids to delete
        """

        raise NotImplementedError

    def search(self, queries, limit):
        """
        Searches ANN index for query. Returns topn results.

        Args:
            queries: queries array
            limit: maximum results

        Returns:
            query results
        """

        raise NotImplementedError

    def count(self):
        """
        Number of elements in the ANN index.

        Returns:
            count
        """

        raise NotImplementedError

    def save(self, path):
        """
        Saves an ANN index at path.

        Args:
            path: path to save ann index
        """

        raise NotImplementedError

    def close(self):
        """
        Closes this ANN.
        """

        self.backend = None

    def setting(self, name, default=None):
        """
        Looks up backend specific setting.

        Args:
            name: setting name
            default: default value when setting not found

        Returns:
            setting value
        """

        # Get the backend-specific config object
        backend = self.config.get(self.config["backend"])

        # Get setting value, set default value if not found
        setting = backend.get(name) if backend else None
        return setting if setting else default

    def metadata(self, settings=None):
        """
        Adds index build metadata.

        Args:
            settings: index build settings
        """

        # ISO 8601 timestamp
        create = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Set build metadata if this is not an update
        if settings:
            self.config["build"] = {
                "create": create,
                "python": platform.python_version(),
                "settings": settings,
                "system": f"{platform.system()} ({platform.machine()})",
                "txtai": __version__,
            }

        # Set last update date
        self.config["update"] = create
