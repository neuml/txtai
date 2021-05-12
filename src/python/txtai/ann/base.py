"""
ANN (Approximate Nearest Neighbors) module
"""


class ANN:
    """
    Base class for ANN models.
    """

    def __init__(self, config):
        """
        Creates a new ANN model.
        """

        # ANN index
        self.model = None

        # Model configuration
        self.config = config

    def load(self, path):
        """
        Loads an ANN model at path.
        """

        raise NotImplementedError

    def index(self, embeddings):
        """
        Builds an ANN model.

        Args:
            embeddings: embeddings array
        """

        raise NotImplementedError

    def append(self, embeddings):
        """
        Append elements to an existing model.

        Args:
            embeddings: embeddings array
        """

        raise NotImplementedError

    def delete(self, ids):
        """
        Deletes elements from existing model.

        Args:
            ids: ids to delete
        """

        raise NotImplementedError

    def search(self, queries, limit):
        """
        Searches ANN model for query. Returns topn results.

        Args:
            queries: queries array
            limit: maximum results

        Returns:
            query results
        """

        raise NotImplementedError

    def count(self):
        """
        Number of elements in the ANN model.

        Returns:
            count
        """

        raise NotImplementedError

    def save(self, path):
        """
        Saves an ANN model at path.
        """

        raise NotImplementedError

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
