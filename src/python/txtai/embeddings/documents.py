"""
Documents module
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
        Creates a new documents stream.
        """

        self.documents = None
        self.batch = 0
        self.size = 0

    def __len__(self):
        """
        Returns total number of queued documents.
        """

        return self.size

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
        # pylint: disable=R1732
        if not self.documents:
            self.documents = tempfile.NamedTemporaryFile(mode="wb", suffix=".docs", delete=False)

        # Add batch
        pickle.dump(documents, self.documents, protocol=4)
        self.batch += 1
        self.size += len(documents)

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
        self.size = 0
