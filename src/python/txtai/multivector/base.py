"""
Multi-vector module
"""


class MultiVector:
    """
    Base class for stored multi-vector components.
    """

    def __init__(self, embeddings, config):
        """
        Creates a multi-vector component.

        Args:
            embeddings: owning embeddings instance
            config: multi-vector configuration
        """

        self.embeddings = embeddings
        self.config = config
        columns = config.get("columns", {})
        self.text = columns.get("text", "text")
        self.object = columns.get("object", "object")

    def insert(self, documents, index):
        """
        Inserts documents at their internal index ids.

        Args:
            documents: list of (id, data, tags)
            index: internal indexid offset
        """

        raise NotImplementedError

    def delete(self, ids):
        """
        Deletes documents at their internal index ids.

        Args:
            ids: internal index ids to delete
        """

        raise NotImplementedError

    def score(self, query, ids, limit=None):
        """
        Scores a query against supplied external candidate ids.

        Args:
            query: query text
            ids: external candidate ids
            limit: maximum results to return, defaults to all

        Returns:
            list of (id, score) sorted descending
        """

        raise NotImplementedError

    def search(self, query, limit=None):
        """
        Searches all documents with stored rows.

        Args:
            query: query text
            limit: maximum results to return, defaults to all

        Returns:
            list of (id, score) sorted descending
        """

        raise NotImplementedError

    def save(self, path):
        """
        Saves the component.

        Args:
            path: output file path
        """

        raise NotImplementedError

    def load(self, path):
        """
        Loads the component.

        Args:
            path: input file path
        """

        raise NotImplementedError

    def close(self):
        """
        Releases component resources.
        """

        raise NotImplementedError
