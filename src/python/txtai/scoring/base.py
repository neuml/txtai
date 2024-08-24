"""
Scoring module
"""


class Scoring:
    """
    Base scoring.
    """

    def __init__(self, config=None):
        """
        Creates a new Scoring instance.

        Args:
            config: input configuration
        """

        # Scoring configuration
        self.config = config if config is not None else {}

        # Transform columns
        columns = self.config.get("columns", {})
        self.text = columns.get("text", "text")
        self.object = columns.get("object", "object")

    def insert(self, documents, index=None):
        """
        Inserts documents into the scoring index.

        Args:
            documents: list of (id, dict|text|tokens, tags)
            index: indexid offset
        """

        raise NotImplementedError

    def delete(self, ids):
        """
        Deletes documents from scoring index.

        Args:
            ids: list of ids to delete
        """

        raise NotImplementedError

    def index(self, documents=None):
        """
        Indexes a collection of documents using a scoring method.

        Args:
            documents: list of (id, dict|text|tokens, tags)
        """

        # Insert documents
        if documents:
            self.insert(documents)

    def upsert(self, documents=None):
        """
        Convience method for API clarity. Calls index method.

        Args:
            documents: list of (id, dict|text|tokens, tags)
        """

        self.index(documents)

    def weights(self, tokens):
        """
        Builds a weights vector for each token in input tokens.

        Args:
            tokens: input tokens

        Returns:
            list of weights for each token
        """

        raise NotImplementedError

    def search(self, query, limit=3):
        """
        Search index for documents matching query.

        Args:
            query: input query
            limit: maximum results

        Returns:
            list of (id, score) or (data, score) if content is enabled
        """

        raise NotImplementedError

    def batchsearch(self, queries, limit=3, threads=True):
        """
        Search index for documents matching queries.

        Args:
            queries: queries to run
            limit: maximum results
            threads: run as threaded search if True and supported
        """

        raise NotImplementedError

    def count(self):
        """
        Returns the total number of documents indexed.

        Returns:
            total number of documents indexed
        """

        raise NotImplementedError

    def load(self, path):
        """
        Loads a saved Scoring object from path.

        Args:
            path: directory path to load scoring index
        """

        raise NotImplementedError

    def save(self, path):
        """
        Saves a Scoring object to path.

        Args:
            path: directory path to save scoring index
        """

        raise NotImplementedError

    def close(self):
        """
        Closes this Scoring object.
        """

        raise NotImplementedError

    def hasterms(self):
        """
        Check if this scoring instance has an associated terms index.

        Returns:
            True if this scoring instance has an associated terms index.
        """

        raise NotImplementedError

    def isnormalized(self):
        """
        Check if this scoring instance returns normalized scores.

        Returns:
            True if normalize is enabled, False otherwise
        """

        raise NotImplementedError
