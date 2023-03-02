"""
Embeddings module
"""

import json
import pickle
import os
import shutil
import tempfile

import numpy as np

from .. import __pickle__

from ..ann import ANNFactory
from ..archive import ArchiveFactory
from ..cloud import CloudFactory
from ..database import DatabaseFactory
from ..graph import GraphFactory
from ..scoring import ScoringFactory
from ..vectors import VectorsFactory

from .explain import Explain
from .functions import Functions
from .reducer import Reducer
from .query import Query
from .search import Search
from .terms import Terms
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

        # Dimensionality reduction and scoring index - word vectors only
        self.reducer, self.scoring = None, None

        # Embeddings vector model - transforms data into similarity vectors
        self.model = None

        # Approximate nearest neighbor index
        self.ann = None

        # Document database
        self.database = None

        # Graph network
        self.graph = None

        # Query model
        self.query = None

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

        # Set configuration to default configuration, if empty
        if not self.config:
            self.configure(self.defaults())

        # Create document database, if necessary
        if not reindex:
            self.database = self.createdatabase()

            # Reset archive since this is a new index
            self.archive = None

        # Create graph, if necessary
        self.graph = self.creategraph()

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

        # Index graph, if necessary
        if self.graph:
            self.graph.index(Search(self, True), self.batchsimilarity)

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
                # Remove principal components from embeddings, if necessary
                if self.reducer:
                    self.reducer(embeddings)

                # Normalize embeddings
                self.normalize(embeddings)

                # Append embeddings to the index
                self.ann.append(embeddings)

                # Save indexids-ids mapping for indexes with no database
                if not self.database:
                    self.config["ids"] = self.config["ids"] + ids

        # Graph upsert, if necessary
        if self.graph:
            self.graph.upsert(Search(self, True))

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

            # Delete ids from graph
            if self.graph:
                self.graph.delete(indices)

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

        return self.batchtransform([document])[0]

    def batchtransform(self, documents, category=None):
        """
        Transforms documents into embeddings vectors.

        Args:
            documents: list of (id, data, tags)
            category: category for instruction-based embeddings

        Returns:
            embeddings vectors
        """

        # Convert documents into sentence embeddings
        embeddings = self.model.batchtransform(documents, category)

        # Reduce the dimensionality of the embeddings. Scale the embeddings using this
        # model to reduce the noise of common but less relevant terms.
        if self.reducer:
            self.reducer(embeddings)

        # Normalize embeddings
        self.normalize(embeddings)

        return embeddings

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

        results = self.batchsearch([query], limit)
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
        queries = self.batchtransform(((None, query, None) for query in queries), "query")
        data = self.batchtransform(((None, row, None) for row in data), "data")

        # Dot product on normalized vectors is equal to cosine similarity
        scores = np.dot(queries, data.T).tolist()

        # Add index and sort desc based on score
        return [sorted(enumerate(score), key=lambda x: x[1], reverse=True) for score in scores]

    def explain(self, query, texts=None, limit=None):
        """
        Explains the importance of each input token in text for a query.

        Args:
            query: input query
            texts: optional list of (text|list of tokens), otherwise runs search query
            limit: optional limit if texts is None

        Returns:
            list of dict per input text where a higher token scores represents higher importance relative to the query
        """

        results = self.batchexplain([query], texts, limit)
        return results[0] if results else results

    def batchexplain(self, queries, texts=None, limit=None):
        """
        Explains the importance of each input token in text for a list of queries.

        Args:
            queries: input queries
            texts: optional list of (text|list of tokens), otherwise runs search queries
            limit: optional limit if texts is None

        Returns:
            list of dict per input text per query where a higher token scores represents higher importance relative to the query
        """

        return Explain(self)(queries, texts, limit)

    def terms(self, query):
        """
        Extracts keyword terms from a query.

        Args:
            query: input query

        Returns:
            query reduced down to keyword terms
        """

        return self.batchterms([query])[0]

    def batchterms(self, queries):
        """
        Extracts keyword terms from a list of queries.

        Args:
            queries: list of queries

        Returns:
            list of queries reduced down to keyword term strings
        """

        return Terms(self)(queries)

    def exists(self, path=None, cloud=None, **kwargs):
        """
        Checks if an index exists at path.

        Args:
            path: input path
            cloud: cloud storage configuration
            kwargs: additional configuration as keyword args

        Returns:
            True if index exists, False otherwise
        """

        # Check if this exists in a cloud instance
        cloud = self.createcloud(cloud=cloud, **kwargs)
        if cloud:
            return cloud.exists(path)

        # Check if this is an archive file and exists
        path, apath = self.checkarchive(path)
        if apath:
            return os.path.exists(apath)

        # Return true if path has a config or config.json file and an embeddings file
        return path and (os.path.exists(f"{path}/config") or os.path.exists(f"{path}/config.json")) and os.path.exists(f"{path}/embeddings")

    def load(self, path=None, cloud=None, **kwargs):
        """
        Loads an existing index from path.

        Args:
            path: input path
            cloud: cloud storage configuration
            kwargs: additional configuration as keyword args
        """

        # Load from cloud, if configured
        cloud = self.createcloud(cloud=cloud, **kwargs)
        if cloud:
            path = cloud.load(path)

        # Check if this is an archive file and extract
        path, apath = self.checkarchive(path)
        if apath:
            self.archive.load(apath)

        # Load index configuration
        self.config = self.loadconfig(path)

        # Approximate nearest neighbor index - stores embeddings vectors
        self.ann = ANNFactory.create(self.config)
        self.ann.load(f"{path}/embeddings")

        # Dimensionality reduction model - word vectors only
        if self.config.get("pca"):
            self.reducer = Reducer()
            self.reducer.load(f"{path}/lsa")

        # Embedding scoring index - word vectors only
        if self.config.get("scoring"):
            self.scoring = ScoringFactory.create(self.config["scoring"])
            self.scoring.load(f"{path}/scoring")

        # Sentence vectors model - transforms data to embeddings vectors
        self.model = self.loadvectors()

        # Query model
        self.query = self.loadquery()

        # Document database - stores document content
        self.database = self.createdatabase()
        if self.database:
            self.database.load(f"{path}/documents")

        # Graph network - stores relationships
        self.graph = self.creategraph()
        if self.graph:
            self.graph.load(f"{path}/graph")

    def save(self, path, cloud=None, **kwargs):
        """
        Saves an index in a directory at path unless path ends with tar.gz, tar.bz2, tar.xz or zip.
        In those cases, the index is stored as a compressed file.

        Args:
            path: output path
            cloud: cloud storage configuration
            kwargs: additional configuration as keyword args
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

            # Save index configuration
            self.saveconfig(path)

            # Save approximate nearest neighbor index
            self.ann.save(f"{path}/embeddings")

            # Save dimensionality reduction model (word vectors only)
            if self.reducer:
                self.reducer.save(f"{path}/lsa")

            # Save embedding scoring index (word vectors only)
            if self.scoring:
                self.scoring.save(f"{path}/scoring")

            # Save document database
            if self.database:
                self.database.save(f"{path}/documents")

            # Save graph
            if self.graph:
                self.graph.save(f"{path}/graph")

            # If this is an archive, save it
            if apath:
                self.archive.save(apath)

            # Save to cloud, if configured
            cloud = self.createcloud(cloud=cloud, **kwargs)
            if cloud:
                cloud.save(apath if apath else path)

    def close(self):
        """
        Closes this embeddings index and frees all resources.
        """

        self.config, self.reducer, self.scoring, self.model = None, None, None, None
        self.ann, self.graph, self.query, self.archive = None, None, None, None

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

        # Query model
        self.query = self.loadquery() if self.config else None

    def defaults(self):
        """
        Builds a default configuration.

        Returns:
            default configuration
        """

        return {"path": "sentence-transformers/all-MiniLM-L6-v2"}

    def loadconfig(self, path):
        """
        Loads index configuration. This method supports both config pickle files and config.json files.

        Args:
            path: path to directory

        Returns:
            dict
        """

        # Configuration
        config = None

        # Determine if config is json or pickle
        jsonconfig = os.path.exists(f"{path}/config.json")

        # Set config file name
        name = "config.json" if jsonconfig else "config"

        # Load configuration
        with open(f"{path}/{name}", "r" if jsonconfig else "rb") as handle:
            config = json.load(handle) if jsonconfig else pickle.load(handle)

        # Build full path to embedding vectors file
        if config.get("storevectors"):
            config["path"] = os.path.join(path, config["path"])

        return config

    def saveconfig(self, path):
        """
        Saves index configuration. This method saves to JSON if possible, otherwise it falls back to pickle.

        Args:
            path: path to directory

        Returns:
            dict
        """

        # Default to pickle config
        jsonconfig = self.config.get("format", "pickle") == "json"

        # Set config file name
        name = "config.json" if jsonconfig else "config"

        # Write configuration
        with open(f"{path}/{name}", "w" if jsonconfig else "wb", encoding="utf-8" if jsonconfig else None) as handle:
            if jsonconfig:
                # Write config as JSON
                json.dump(self.config, handle, default=str, indent=2)
            else:
                # Write config as pickle format
                pickle.dump(self.config, handle, protocol=__pickle__)

    def loadvectors(self):
        """
        Loads a vector model set in config.

        Returns:
            vector model
        """

        return VectorsFactory.create(self.config, self.scoring)

    def loadquery(self):
        """
        Loads a query model set in config.

        Returns:
            query model
        """

        if "query" in self.config:
            return Query(**self.config["query"])

        return None

    def checkarchive(self, path):
        """
        Checks if path is an archive file.

        Args:
            path: path to check

        Returns:
            (working directory, current path) if this is an archive, original path otherwise
        """

        # Create archive instance, if necessary
        self.archive = ArchiveFactory.create()

        # Check if path is an archive file
        if self.archive.isarchive(path):
            # Return temporary archive working directory and original path
            return self.archive.path(), path

        return path, None

    def createcloud(self, **cloud):
        """
        Creates a cloud instance from config.

        Args:
            cloud: cloud configuration
        """

        # Merge keyword args and keys under the cloud parameter
        config = cloud
        if "cloud" in config and config["cloud"]:
            config.update(config.pop("cloud"))

        # Create cloud instance from config and return
        return CloudFactory.create(config) if config else None

    def createdatabase(self):
        """
        Creates a database from config. This method will also close any existing database connection.

        Returns:
            new database, if enabled in config
        """

        # Free existing database resources
        if self.database:
            self.database.close()

        config = self.config.copy()

        # Resolve callable functions
        if "functions" in config:
            config["functions"] = Functions(self)(config)

        # Create database from config and return
        return DatabaseFactory.create(config)

    def creategraph(self):
        """
        Creates a graph from config.

        Returns:
            new graph, if enabled in config
        """

        return GraphFactory.create(self.config["graph"]) if "graph" in self.config else None

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
