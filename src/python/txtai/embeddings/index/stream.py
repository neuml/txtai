"""
Stream module
"""

from .autoid import AutoId
from .transform import Action


class Stream:
    """
    Yields input document as standard (id, data, tags) tuples.
    """

    def __init__(self, embeddings, action=None):
        """
        Create a new stream.

        Args:
            embeddings: embeddings instance
            action: optional index action
        """

        self.embeddings = embeddings
        self.action = action

        # Alias embeddings attributes
        self.config = embeddings.config

        # Get config parameters
        self.offset = self.config.get("offset", 0) if action == Action.UPSERT else 0
        autoid = self.config.get("autoid", self.offset)

        # Create autoid generator, reset int sequence if this isn't an UPSERT
        autoid = 0 if isinstance(autoid, int) and action != Action.UPSERT else autoid
        self.autoid = AutoId(autoid)

    def __call__(self, documents):
        """
        Yield (id, data, tags) tuples from a stream of documents.

        Args:
            documents: input documents
        """

        # Iterate over documents and yield standard (id, data, tag) tuples
        for document in documents:
            if isinstance(document, dict):
                # Create (id, data, tags) tuple from dictionary
                document = document.get("id"), document, document.get("tags")
            elif isinstance(document, tuple):
                # Create (id, data, tags) tuple
                document = document if len(document) >= 3 else (document[0], document[1], None)
            else:
                # Create (id, data, tags) tuple with empty fields
                document = None, document, None

            # Set autoid if the action is set
            if self.action and document[0] is None:
                document = (self.autoid(document[1]), document[1], document[2])

            # Yield (id, data, tags) tuple
            yield document

        # Save autoid sequence if used
        current = self.autoid.current()
        if self.action and current:
            self.config["autoid"] = current
