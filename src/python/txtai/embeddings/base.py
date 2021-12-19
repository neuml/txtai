"""
Embeddings module
"""

import pickle
import os
import shutil

import numpy as np

from ..ann import ANNFactory
from ..database import DatabaseFactory
from ..scoring import ScoringFactory
from ..vectors import VectorsFactory

from .archive import Archive
from .reducer import Reducer
from .search import Search

# pylint: disable=R0904
class Embeddings:
    """
    Embeddings is the engine that delivers semantic search. Text is transformed into embeddings vectors where similar concepts
    will produce similar vectors. Indexes both large and small are built with these vectors. The indexes are used find results
    that have the same meaning, not necessarily the same keywords.
    """

    # pylint: disable = W0231
    def __init__(self, config=None):
        """
        Creates a new embeddings index. Embeddings indexes are thread-safe for read operations but writes must be
        synchronized.

        Args:
            config: embeddings configuration
        """

        # Index configuration
        self.config = None

        # Dimensionality reduction and scoring models - word vectors only
        self.reducer, self.scoring = None, None

        # Embeddings vector model - transforms text into similarity vectors
        self.model = None

        # Approximate nearest neighbor index
        self.ann = None

        # Document database
        self.database = None

        # Index archive
        self.archive = None

        # Set initial configuration
        self.configure(config)

    def score(self, documents):
        """
        Builds a scoring index. Only used by word vectors models.

        Args:
            documents: list of (id, dict|text|tokens, tags)
        """

        # Build scoring index over documents
        if self.scoring:
            self.scoring.index(documents)

    def index(self, documents, reindex=False):
        """
        Builds an embeddings index. This method overwrites an existing index.

        Args:
            documents: list of (id, dict|text|tokens, tags)
            reindex: if this is a reindex operation in which case database creation is skipped, defaults to False
        """

        # Transform documents to embeddings vectors
        ids, dimensions, embeddings = self.vectors(documents)

        # Build LSA model (if enabled). Remove principal components from embeddings.
        if self.config.get("pca"):
            self.reducer = Reducer(embeddings, self.config["pca"])
            self.reducer(embeddings)

        # Normalize embeddings
        self.normalize(embeddings)

        # Save index dimensions
        self.config["dimensions"] = dimensions

        # Create approximate nearest neighbor index
        self.ann = ANNFactory.create(self.config)

        # Build the index
        self.ann.index(embeddings)

        # Keep existing database and archive instances if this is part of a reindex
        if not reindex:
            # Create document database
            self.database = self.createdatabase()
            if self.database:
                # Add documents to database
                self.database.insert(documents)
            else:
                # Save indexids-ids mapping for indexes with no database
                self.config["ids"] = ids

            # Reset archive since this is a new index
            self.archive = None

    def upsert(self, documents):
        """
        Runs an embeddings upsert operation. If the index exists, new data is
        appended to the index, existing data is updated. If the index doesn't exist,
        this method runs a standard index operation.

        Args:
            documents: list of (id, dict|text|tokens, tags)
        """

        # Run standard insert if index doesn't exist
        if not self.ann:
            self.index(documents)
            return

        # Transform documents to embeddings vectors
        ids, _, embeddings = self.vectors(documents)

        # Normalize embeddings
        self.normalize(embeddings)

        # Delete existing elements
        self.delete(ids)

        # Get offset before it changes
        offset = self.config.get("offset", 0)

        # Append elements the index
        self.ann.append(embeddings)

        if self.database:
            # Add documents to database
            self.database.insert(documents, offset)
        else:
            # Save indexids-ids mapping for indexes with no database
            self.config["ids"] = self.config["ids"] + ids

    def delete(self, ids):
        """
        Deletes from an embeddings index. Returns list of ids deleted.

        Args:
            ids: list of ids to delete

        Returns:
            list of ids deleted
        """

        # List of internal indices for each candidate id to delete
        indices = []

        # List of deleted ids
        deletes = []

        if self.database:
            # Retrieve indexid-id mappings from database
            ids = self.database.ids(ids)

            # Parse out indices and ids to delete
            indices = [i for i, _ in ids]
            deletes = sorted(set(uid for _, uid in ids))

            # Delete ids from database
            self.database.delete(deletes)
        else:
            # Lookup indexids from config for indexes with no database
            indexids = self.config["ids"]

            # Find existing ids
            for uid in ids:
                indices.extend([index for index, value in enumerate(indexids) if uid == value])

            # Clear config ids
            for index in indices:
                deletes.append(indexids[index])
                indexids[index] = None

        # Delete indices from ann embeddings
        if indices:
            # Delete ids from index
            self.ann.delete(indices)

        return deletes

    def reindex(self, config, columns=None):
        """
        Recreates the approximate nearest neighbor (ann) index using config. This method only works if document
        content storage is enabled.

        Args:
            config: new config
            columns: optional list of document columns used to rebuild text
        """

        if self.database:
            # Keep content parameter to ensure database is preserved
            config["content"] = self.config["content"]

            # Reset configuration
            self.configure(config)

            # Reindex
            self.index(self.database.reindex(columns), True)

    def transform(self, document):
        """
        Transforms document into an embeddings vector. Document text will be tokenized if not pre-tokenized.

        Args:
            document: (id, text|tokens, tags)

        Returns:
            embeddings vector
        """

        # Convert document into sentence embedding
        embedding = self.model.transform(document)

        # Reduce the dimensionality of the embeddings. Scale the embeddings using this
        # model to reduce the noise of common but less relevant terms.
        if self.reducer:
            self.reducer(embedding)

        # Normalize embeddings
        self.normalize(embedding)

        return embedding

    def batchtransform(self, documents):
        """
        Transforms documents into embeddings vectors. Document text will be tokenized if not pre-tokenized.

        Args:
            documents: list of (id, text|tokens, tags)

        Returns:
            embeddings vectors
        """

        return [self.transform(document) for document in documents]

    def count(self):
        """
        Total number of elements in this embeddings index.

        Returns:
            number of elements in this embeddings index
        """

        return self.ann.count() if self.ann else 0

    def search(self, query, limit=3):
        """
        Finds documents most similar to the input queries. This method will run either an approximate
        nearest neighbor (ann) search or an approximate nearest neighbor + database search depending
        on if a database is available.

        Args:
            query: query text|tokens
            limit: maximum results

        Returns:
            list of (id, score) for ann search, list of dict for an ann+database search
        """

        return self.batchsearch([query], limit)[0]

    def batchsearch(self, queries, limit=3):
        """
        Finds documents most similar to the input queries. This method will run either an approximate
        nearest neighbor (ann) search or an approximate nearest neighbor + database search depending
        on if a database is available.

        Args:
            queries: queries text|tokens
            limit: maximum results

        Returns:
            list of (id, score) per query for ann search, list of dict per query for an ann+database search
        """

        return Search(self)(queries, limit)

    def similarity(self, query, texts):
        """
        Computes the similarity between query and list of text. Returns a list of
        (id, score) sorted by highest score, where id is the index in texts.

        Args:
            query: query text|tokens
            texts: list of text|tokens

        Returns:
            list of (id, score)
        """

        return self.batchsimilarity([query], texts)[0]

    def batchsimilarity(self, queries, texts):
        """
        Computes the similarity between list of queries and list of text. Returns a list
        of (id, score) sorted by highest score per query, where id is the index in texts.

        Args:
            queries: queries text|tokens
            texts: list of text|tokens

        Returns:
            list of (id, score) per query
        """

        # Convert queries to embedding vectors
        queries = np.array([self.transform((None, query, None)) for query in queries])
        texts = np.array([self.transform((None, text, None)) for text in texts])

        # Dot product on normalized vectors is equal to cosine similarity
        scores = np.dot(queries, texts.T).tolist()

        # Add index and sort desc based on score
        return [sorted(enumerate(score), key=lambda x: x[1], reverse=True) for score in scores]

    def load(self, path):
        """
        Loads an existing index from path.

        Args:
            path: input path
        """

        # Check if this is an archive file and extract
        path, apath = self.checkarchive(path)
        if apath:
            self.archive.load(apath)

        # Index configuration
        with open(f"{path}/config", "rb") as handle:
            self.config = pickle.load(handle)

            # Build full path to embedding vectors file
            if self.config.get("storevectors"):
                self.config["path"] = os.path.join(path, self.config["path"])

        # Approximate nearest neighbor index - stores embeddings vectors
        self.ann = ANNFactory.create(self.config)
        self.ann.load(f"{path}/embeddings")

        # Dimensionality reduction model - word vectors only
        if self.config.get("pca"):
            self.reducer = Reducer()
            self.reducer.load(f"{path}/lsa")

        # Embedding scoring model - word vectors only
        if self.config.get("scoring"):
            self.scoring = ScoringFactory.create(self.config["scoring"])
            self.scoring.load(f"{path}/scoring")

        # Sentence vectors model - transforms text to embeddings vectors
        self.model = self.loadvectors()

        # Document database - stores document content
        self.database = self.createdatabase()
        if self.database:
            self.database.load(f"{path}/documents")

    def exists(self, path):
        """
        Checks if an index exists at path.

        Args:
            path: input path

        Returns:
            True if index exists, False otherwise
        """

        return os.path.exists(f"{path}/config") and os.path.exists(f"{path}/embeddings")

    def save(self, path):
        """
        Saves an index.

        Args:
            path: output path
        """

        if self.config:
            # Check if this is an archive file
            path, apath = self.checkarchive(path)

            # Create output directory, if necessary
            os.makedirs(path, exist_ok=True)

            # Copy sentence vectors model
            if self.config.get("storevectors"):
                shutil.copyfile(self.config["path"], os.path.join(path, os.path.basename(self.config["path"])))

                self.config["path"] = os.path.basename(self.config["path"])

            # Write index configuration
            with open(f"{path}/config", "wb") as handle:
                pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Save approximate nearest neighbor index
            self.ann.save(f"{path}/embeddings")

            # Save dimensionality reduction model (word vectors only)
            if self.reducer:
                self.reducer.save(f"{path}/lsa")

            # Save embedding scoring model (word vectors only)
            if self.scoring:
                self.scoring.save(f"{path}/scoring")

            # Save document database
            if self.database:
                self.database.save(f"{path}/documents")

            # If this is an archive, save it
            if apath:
                self.archive.save(apath)

    def close(self):
        """
        Closes this embeddings index and frees all resources.
        """

        self.config, self.reducer, self.scoring, self.model, self.ann, self.archive = None, None, None, None, None, None

        # Close database connection if open
        if self.database:
            self.database.close()
            self.database = None

    def configure(self, config):
        """
        Sets the configuration for this embeddings index and loads config-driven models.

        Args:
            config: embeddings configuration
        """

        # Configuration
        self.config = config

        if self.config and self.config.get("method") != "transformers":
            # Dimensionality reduction model
            self.reducer = None

            # Embedding scoring method - weighs each word in a sentence
            self.scoring = ScoringFactory.create(self.config["scoring"]) if self.config and self.config.get("scoring") else None
        else:
            self.reducer, self.scoring = None, None

        # Sentence vectors model - transforms text to embeddings vectors
        self.model = self.loadvectors() if self.config else None

    def loadvectors(self):
        """
        Loads a vector model set in config.

        Returns:
            vector model
        """

        return VectorsFactory.create(self.config, self.scoring)

    def vectors(self, documents):
        """
        Transforms documents into embeddings vectors.

        Args:
            documents: list of (id, dict|text|tokens, tags)

        Returns:
            (document ids, dimensions, embeddings)
        """

        # Transform documents to embeddings vectors
        ids, dimensions, stream = self.model.index(self.gettext(documents))

        # Load streamed embeddings back to memory
        embeddings = np.empty((len(ids), dimensions), dtype=np.float32)
        with open(stream, "rb") as queue:
            for x in range(embeddings.shape[0]):
                embeddings[x] = pickle.load(queue)

        # Remove temporary file
        os.remove(stream)

        return (ids, dimensions, embeddings)

    def gettext(self, documents):
        """
        Selects documents that have text to vectorize for similarity indexing.

        Must be one of the following:
            - text
            - list of text tokens
            - dict with the field "text"

        Args:
            documents: list of (id, dict|text|tokens, tags)

        Returns:
            filtered documents
        """

        for document in documents:
            if isinstance(document[1], (str, list)):
                yield document
            elif isinstance(document[1], dict) and "text" in document[1]:
                yield (document[0], document[1]["text"], document[2])

    def checkarchive(self, path):
        """
        Checks if path is an archive file.

        Args:
            path: path to check

        Returns:
            (working directory, current path) if this is an archive, original path otherwise
        """

        # Create archive instance, if necessary
        self.archive = self.archive if self.archive else Archive()

        # Check if path is an archive file
        if self.archive.isarchive(path):
            # Return temporary archive working directory and original path
            return self.archive.path(), path

        return path, None

    def createdatabase(self):
        """
        Creates a database from config. This method will also close any existing database connection.

        Returns:
            new database, if enabled in config
        """

        # Free existing database resources
        if self.database:
            self.database.close()

        # Create database from config and return
        return DatabaseFactory.create(self.config)

    def normalize(self, embeddings):
        """
        Normalizes embeddings using L2 normalization. Operation applied directly on array.

        Args:
            embeddings: input embeddings matrix
        """

        # Calculation is different for matrices vs vectors
        if len(embeddings.shape) > 1:
            embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        else:
            embeddings /= np.linalg.norm(embeddings)
