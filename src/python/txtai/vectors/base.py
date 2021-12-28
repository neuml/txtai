"""
Vectors module
"""


class Vectors:
    """
    Base class for sentence embeddings/vector models.
    """

    def __init__(self, config, scoring):
        # Store parameters
        self.config = config
        self.scoring = scoring

        if config:
            # Detect if this is an initialized configuration
            self.initialized = "dimensions" in config

            # Enables optional string tokenization
            self.tokenize = config.get("tokenize")

            # pylint: disable=E1111
            self.model = self.load(config.get("path"))

    def load(self, path):
        """
        Loads vector model at path.

        Args:
            path: path to vector model

        Returns:
            vector model
        """

        raise NotImplementedError

    def index(self, documents):
        """
        Converts a list of documents to a temporary file with embeddings arrays. Returns a tuple of document ids,
        number of dimensions and temporary file with embeddings.

        Args:
            documents: list of (id, data, tags)

        Returns:
            (ids, dimensions, stream)
        """

        raise NotImplementedError

    def transform(self, document):
        """
        Transforms document into an embeddings vector.

        Args:
            document: (id, data, tags)

        Returns:
            embeddings vector
        """

        raise NotImplementedError
