"""
Embeddings module
"""

import json
import pickle
import os
import shutil
import tempfile

import numpy as np

from ..ann import ANNFactory
from ..database import DatabaseFactory
from ..scoring import ScoringFactory
from ..vectors import VectorsFactory

from .archive import Archive
from .reducer import Reducer
from .search import Search
from .transform import Action, Transform


# pylint: disable=R0904
class Embeddings:
    """
    Embeddings is the engine that delivers semantic search. Data is transformed into embeddings vectors where similar concepts
    will produce similar vectors. Indexes both large and small are built with these vectors. The indexes are used to find results
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

        # Embeddings vector model - transforms data into similarity vectors
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
            documents: list of (id, data, tags)
        """

        # Build scoring index over documents
        if self.scoring:
            self.scoring.index(documents)

    def index(self, documents, reindex=False):
        """
        Builds an embeddings index. This method overwrites an existing index.

        Args:
            documents: list of (id, data, tags)
            reindex: if this is a reindex operation in which case database creation is skipped, defaults to False
        """

        # Create document database, if necessary
        if not reindex:
            self.database = self.createdatabase()

            # Reset archive since this is a new index
            self.archive = None

        # Create transform action
        transform = Transform(self, Action.REINDEX if reindex else Action.INDEX)

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy") as buffer:
            # Load documents into database and transform to vectors
            ids, dimensions, embeddings = transform(documents, buffer)
            if ids:
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

                # Add embeddings to the index
                self.ann.index(embeddings)

                # Save indexids-ids mapping for indexes with no database, except when this is a reindex action
                if not reindex and not self.database:
                    self.config["ids"] = ids

    def upsert(self, documents):
        """
        Runs an embeddings upsert operation. If the index exists, new data is
        appended to the index, existing data is updated. If the index doesn't exist,
        this method runs a standard index operation.

        Args:
            documents: list of (id, data, tags)
        """

        # Run standard insert if index doesn't exist
        if not self.ann:
            self.index(documents)
            return

        # Create transform action
        transform = Transform(self, Action.UPSERT)

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy") as buffer:
            # Load documents into database and transform to vectors
            ids, _, embeddings = transform(documents, buffer)
            if ids:
                # Normalize embeddings
                self.normalize(embeddings)

                # Append embeddings to the index
                self.ann.append(embeddings)

                # Save indexids-ids mapping for indexes with no database
                if not self.database:
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
        elif self.ann:
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

    def reindex(self, config, columns=None, function=None):
        """
        Recreates the approximate nearest neighbor (ann) index using config. This method only works if document
        content storage is enabled.

        Args:
            config: new config
            columns: optional list of document columns used to rebuild data
            function: optional function to prepare content for indexing
        """

        if self.database:
            # Keep content and objects parameters to ensure database is preserved
            config["content"] = self.config["content"]
            if "objects" in self.config:
                config["objects"] = self.config["objects"]

            # Reset configuration
            self.configure(config)

            # Reindex
            if function:
                self.index(function(self.database.reindex(columns)), True)
            else:
                self.index(self.database.reindex(columns), True)

    def transform(self, document):
        """
        Transforms document into an embeddings vector.

        Args:
            document: (id, data, tags)

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
        Transforms documents into embeddings vectors.

        Args:
            documents: list of (id, data, tags)

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

    def search(self, query, limit=None):
        """
        Finds documents most similar to the input queries. This method will run either an approximate
        nearest neighbor (ann) search or an approximate nearest neighbor + database search depending
        on if a database is available.

        Args:
            query: input query
            limit: maximum results

        Returns:
            list of (id, score) for ann search, list of dict for an ann+database search
        """

        results = self.batchsearch([query], limit if limit else 3)
        return results[0] if results else results

    def batchsearch(self, queries, limit=None):
        """
        Finds documents most similar to the input queries. This method will run either an approximate
        nearest neighbor (ann) search or an approximate nearest neighbor + database search depending
        on if a database is available.

        Args:
            queries: input queries
            limit: maximum results

        Returns:
            list of (id, score) per query for ann search, list of dict per query for an ann+database search
        """

        return Search(self)(queries, limit if limit else 3)

    def similarity(self, query, data):
        """
        Computes the similarity between query and list of data. Returns a list of
        (id, score) sorted by highest score, where id is the index in data.

        Args:
            query: input query
            data: list of data

        Returns:
            list of (id, score)
        """

        return self.batchsimilarity([query], data)[0]

    def batchsimilarity(self, queries, data):
        """
        Computes the similarity between list of queries and list of data. Returns a list
        of (id, score) sorted by highest score per query, where id is the index in data.

        Args:
            queries: input queries
            data: list of data

        Returns:
            list of (id, score) per query
        """

        # Convert queries to embedding vectors
        queries = np.array([self.transform((None, query, None)) for query in queries])
        data = np.array([self.transform((None, row, None)) for row in data])

        # Dot product on normalized vectors is equal to cosine similarity
        scores = np.dot(queries, data.T).tolist()

        # Add index and sort desc based on score
        return [sorted(enumerate(score), key=lambda x: x[1], reverse=True) for score in scores]

    def exists(self, path, cloud=None):
        """
        Checks if an index exists at path.

        Args:
            path: input path
            cloud: cloud storage configuration

        Returns:
            True if index exists, False otherwise
        """

        # Check if this is an archive file and exists
        path, apath = self.checkarchive(path)
        if apath:
            return self.archive.exists(apath, cloud)

        return os.path.exists(f"{path}/config") and os.path.exists(f"{path}/embeddings")

    def load(self, path, cloud=None):
        """
        Loads an existing index from path.

        Args:
            path: input path
            cloud: cloud storage configuration
        """

        # Check if this is an archive file and extract
        path, apath = self.checkarchive(path)
        if apath:
            self.archive.load(apath, cloud)

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

        # Sentence vectors model - transforms data to embeddings vectors
        self.model = self.loadvectors()

        # Document database - stores document content
        self.database = self.createdatabase()
        if self.database:
            self.database.load(f"{path}/documents")

    def save(self, path, cloud=None):
        """
        Saves an index.

        Args:
            path: output path
            cloud: cloud storage configuration
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
                pickle.dump(self.config, handle, protocol=4)

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
                self.archive.save(apath, cloud)

    def close(self):
        """
        Closes this embeddings index and frees all resources.
        """

        self.config, self.reducer, self.scoring, self.model, self.ann, self.archive = None, None, None, None, None, None

        # Close database connection if open
        if self.database:
            self.database.close()
            self.database = None

    def info(self):
        """
        Prints the current embeddings index configuration.
        """

        # Copy and edit config
        config = self.config.copy()

        # Remove ids array if present
        config.pop("ids", None)

        # Print configuration
        print(json.dumps(config, sort_keys=True, default=str, indent=2))

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

        # Sentence vectors model - transforms data to embeddings vectors
        self.model = self.loadvectors() if self.config else None

    def loadvectors(self):
        """
        Loads a vector model set in config.

        Returns:
            vector model
        """

        return VectorsFactory.create(self.config, self.scoring)

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
