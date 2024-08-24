"""
Indexes module
"""

import os

from .documents import Documents


class Indexes:
    """
    Manages a collection of subindexes for an embeddings instance.
    """

    def __init__(self, embeddings, indexes):
        """
        Creates a new indexes instance.

        Args:
            embeddings: embeddings instance
            indexes: dict of subindexes to add
        """

        self.embeddings = embeddings
        self.indexes = indexes

        self.documents = None

        # Transform columns
        columns = embeddings.config.get("columns", {})
        self.text = columns.get("text", "text")
        self.object = columns.get("object", "object")

        # Check if top-level indexing is enabled for this embeddings instance
        self.indexing = embeddings.model or embeddings.scoring

    def __contains__(self, name):
        """
        Returns True if name is in this instance, False otherwise.

        Returns:
            True if name is in this instance, False otherwise
        """

        return name in self.indexes

    def __getitem__(self, name):
        """
        Looks up an index by name.

        Args:
            name: index name

        Returns:
            index
        """

        return self.indexes[name]

    def __getattr__(self, name):
        """
        Looks up an index by attribute name.

        Args:
            name: index name

        Returns:
            index
        """

        try:
            return self.indexes[name]
        except Exception as e:
            raise AttributeError(e) from e

    def default(self):
        """
        Gets the default/first index.

        Returns:
            default index
        """

        return list(self.indexes.keys())[0]

    def model(self, index=None):
        """
        Lookups a vector model. If index is empty, the first vector model is returned.

        Returns:
            Vectors
        """

        # Find vector model
        matches = [self.indexes[index]] if index else [index.model for index in self.indexes.values() if index.model]
        return matches[0] if matches else None

    def insert(self, documents, index=None):
        """
        Inserts a batch of documents into each subindex.

        Args:
            documents: list of (id, data, tags)
            index: indexid offset
        """

        if not self.documents:
            self.documents = Documents()

        # Create batch containing documents added to parent index
        batch = []
        for _, document, _ in documents:
            # Add to documents collection if text or object field is set
            parent = document
            if isinstance(parent, dict):
                parent = parent.get(self.text, document.get(self.object))

            # Add if field is available or top-level indexing is disabled
            if parent is not None or not self.indexing:
                batch.append((index, document, None))
                index += 1

        # Add filtered documents batch
        self.documents.add(batch)

    def delete(self, ids):
        """
        Deletes ids from each subindex.

        Args:
            ids: list of ids to delete
        """

        for index in self.indexes.values():
            index.delete(ids)

    def index(self):
        """
        Builds each subindex.
        """

        for index in self.indexes.values():
            index.index(self.documents)

        # Reset document stream
        self.documents.close()
        self.documents = None

    def upsert(self):
        """
        Runs upsert for each subindex.
        """

        for index in self.indexes.values():
            index.upsert(self.documents)

        # Reset document stream
        self.documents.close()
        self.documents = None

    def load(self, path):
        """
        Loads each subindex from path.

        Args:
            path: directory path to load subindexes
        """

        for name, index in self.indexes.items():
            # Load subindex if it exists, subindexes aren't required to have data
            directory = os.path.join(path, name)
            if index.exists(directory):
                index.load(directory)

    def save(self, path):
        """
        Saves each subindex to path.

        Args:
            path: directory path to save subindexes
        """

        for name, index in self.indexes.items():
            index.save(os.path.join(path, name))

    def close(self):
        """
        Close and free resources used by this instance.
        """

        for index in self.indexes.values():
            index.close()
