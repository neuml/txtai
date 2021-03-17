"""
Stream module
"""

import os
import pickle
import tempfile


class Documents:
    """
    Streams documents to temporary storage. Allows queuing large volumes of content for later indexing.
    """

    def __init__(self):
        """
        Creates a new DocumentStream.
        """

        self.documents = None
        self.batch = 0

    def __iter__(self):
        """
        Streams all queued documents.
        """

        # Close streaming file
        self.documents.close()

        # Open stream file
        with open(self.documents.name, "rb") as queue:
            # Read each batch
            for _ in range(self.batch):
                documents = pickle.load(queue)

                # Yield each document
                yield from documents

    def add(self, documents):
        """
        Adds a batch of documents for indexing.

        Args:
            documents: list of (id, data, tag) tuples

        Returns:
            documents
        """

        # Create documents file if not already open
        if not self.documents:
            self.documents = tempfile.NamedTemporaryFile(mode="wb", suffix=".docs", delete=False)

        # Add batch
        pickle.dump(documents, self.documents)
        self.batch += 1

        return documents

    def close(self):
        """
        Closes and resets this instance. New sets of documents can be added with additional calls to add.
        """

        # Cleanup stream file
        os.remove(self.documents.name)

        # Reset document parameters
        self.documents = None
        self.batch = 0
