"""
Transform module
"""

import os
import pickle

from enum import Enum

import numpy as np


class Transform:
    """
    Executes a transform action. Processes a stream of documents, loads batches into a database and vectorizes documents.
    """

    def __init__(self, embeddings, action):
        """
        Creates a new transform action.

        Args:
            embeddings: embeddings instance
            action: transform action
        """

        self.embeddings = embeddings
        self.action = action

        # Alias embeddings attributes
        self.delete = self.embeddings.delete
        self.model = self.embeddings.model
        self.database = self.embeddings.database

        # Get config parameters
        self.offset = self.embeddings.config.get("offset", 0) if self.action == Action.UPSERT else 0
        self.batch = self.embeddings.config.get("batch", 1024)

        # List of deleted ids with this action
        self.deletes = set()

    def __call__(self, documents, buffer):
        """
        Processes an iterable collection of documents, handles any iterable including generators.

        This method loads a stream of documents into a database (if applicable) and vectorizes documents into an embeddings array.

        Args:
            documents: iterable collection of (id, data, tags)
            buffer: file path used for memmap buffer

        Returns:
            (document ids, dimensions, embeddings)
        """

        # Transform documents to vectors and load into database
        ids, dimensions, batches, stream = self.model.index(self.stream(documents))

        # Check that embeddings are available and load as a memmap
        embeddings = None
        if ids:
            embeddings = np.memmap(buffer, dtype=np.float32, shape=(len(ids), dimensions), mode="w+")
            with open(stream, "rb") as queue:
                x = 0
                for _ in range(batches):
                    batch = pickle.load(queue)
                    embeddings[x : x + batch.shape[0]] = batch
                    x += batch.shape[0]

        # Remove temporary file
        os.remove(stream)

        return (ids, dimensions, embeddings)

    def stream(self, documents):
        """
        This method does two things:

        1. Filter and yield data to vectorize
        2. Batch and load original documents into a database (if applicable)

        Documents are yielded for vectorization if one of the following is True:
            - dict with the field "text" or "object"
            - not a dict

        Otherwise, documents are only batched and inserted into a database

        Args:
            documents: iterable collection (id, data, tags)
        """

        batch = []

        # Iterate and process documents stream
        for document in documents:
            if isinstance(document[1], dict):
                if "text" in document[1]:
                    yield (document[0], document[1]["text"], document[2])
                elif "object" in document[1]:
                    yield (document[0], document[1]["object"], document[2])
            else:
                yield document

            # Batch document
            batch.append(document)
            if len(batch) == self.batch:
                self.load(batch)
                batch = []

        # Final batch
        if batch:
            self.load(batch)

    def load(self, batch):
        """
        Loads a document batch. This method deletes existing ids from an embeddings index and
        load into a database, if applicable.

        Args:
            batch: list of (id, data, tags)
        """

        # Delete from embeddings index first (which deletes from underlying ANN index and database) if this is an upsert
        if self.action == Action.UPSERT:
            # Get list of ids not yet seen and delete
            deletes = [uid for uid, _, _ in batch if uid not in self.deletes]
            if deletes:
                # Execute delete
                self.delete(deletes)

                # Save deleted ids as a delete must only occur once per action
                self.deletes.update(deletes)

        # Load batch into database except if this is a reindex
        if self.database and self.action != Action.REINDEX:
            self.database.insert(batch, self.offset)
            self.offset += len(batch)


class Action(Enum):
    """
    Transform action types
    """

    INDEX = 1
    UPSERT = 2
    REINDEX = 3
