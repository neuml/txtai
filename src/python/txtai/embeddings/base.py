"""
Embeddings module
"""

import json
import os
import tempfile

import numpy as np

from ..ann import ANNFactory
from ..archive import ArchiveFactory
from ..cloud import CloudFactory
from ..database import DatabaseFactory
from ..graph import GraphFactory
from ..scoring import ScoringFactory
from ..vectors import VectorsFactory

from .index import Action, Configuration, Functions, Indexes, IndexIds, Reducer, Stream, Transform
from .search import Explain, Ids, Query, Search, Terms


# pylint: disable=C0302,R0904
class Embeddings:
    """
    Embeddings databases are the engine that delivers semantic search. Data is transformed into embeddings vectors where similar concepts
    will produce similar vectors. Indexes both large and small are built with these vectors. The indexes are used to find results
    that have the same meaning, not necessarily the same keywords.
    """

    # pylint: disable = W0231
    def __init__(self, config=None, models=None, **kwargs):
        """
        Creates a new embeddings index. Embeddings indexes are thread-safe for read operations but writes must be synchronized.

        Args:
            config: embeddings configuration
            models: models cache, used for model sharing between embeddings
            kwargs: additional configuration as keyword args
        """

        # Index configuration
        self.config = None

        # Dimensionality reduction - word vectors only
        self.reducer = None

        # Dense vector model - transforms data into similarity vectors
        self.model = None

        # Approximate nearest neighbor index
        self.ann = None

        # Index ids when content is disabled
        self.ids = None

        # Document database
        self.database = None

        # Resolvable functions
        self.functions = None

        # Graph network
        self.graph = None

        # Sparse vectors
        self.scoring = None

        # Query model
        self.query = None

        # Index archive
        self.archive = None

        # Subindexes for this embeddings instance
        self.indexes = None

        # Models cache
        self.models = models

        # Merge configuration into single dictionary
        config = {**config, **kwargs} if config and kwargs else kwargs if kwargs else config

        # Set initial configuration
        self.configure(config)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def score(self, documents):
        """
        Builds a term weighting scoring index. Only used by word vectors models.

        Args:
            documents: iterable of (id, data, tags), (id, data) or data
        """

        # Build scoring index for word vectors term weighting
        if self.isweighted():
            self.scoring.index(Stream(self)(documents))

    def index(self, documents, reindex=False, checkpoint=None):
        """
        Builds an embeddings index. This method overwrites an existing index.

        Args:
            documents: iterable of (id, data, tags), (id, data) or data
            reindex: if this is a reindex operation in which case database creation is skipped, defaults to False
            checkpoint: optional checkpoint directory, enables indexing restart
        """

        # Initialize index
        self.initindex(reindex)

        # Create transform and stream
        transform = Transform(self, Action.REINDEX if reindex else Action.INDEX, checkpoint)
        stream = Stream(self, Action.REINDEX if reindex else Action.INDEX)

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy") as buffer:
            # Load documents into database and transform to vectors
            ids, dimensions, embeddings = transform(stream(documents), buffer)
            if embeddings is not None:
                # Build LSA model (if enabled). Remove principal components from embeddings.
                if self.config.get("pca"):
                    self.reducer = Reducer(embeddings, self.config["pca"])
                    self.reducer(embeddings)

                # Save index dimensions
                self.config["dimensions"] = dimensions

                # Create approximate nearest neighbor index
                self.ann = self.createann()

                # Add embeddings to the index
                self.ann.index(embeddings)

            # Save indexids-ids mapping for indexes with no database, except when this is a reindex
            if ids and not reindex and not self.database:
                self.ids = self.createids(ids)

        # Index scoring, if necessary
        # This must occur before graph index in order to be available to the graph
        if self.issparse():
            self.scoring.index()

        # Index subindexes, if necessary
        if self.indexes:
            self.indexes.index()

        # Index graph, if necessary
        if self.graph:
            self.graph.index(Search(self, indexonly=True), Ids(self), self.batchsimilarity)

    def upsert(self, documents, checkpoint=None):
        """
        Runs an embeddings upsert operation. If the index exists, new data is
        appended to the index, existing data is updated. If the index doesn't exist,
        this method runs a standard index operation.

        Args:
            documents: iterable of (id, data, tags), (id, data) or data
            checkpoint: optional checkpoint directory, enables indexing restart
        """

        # Run standard insert if index doesn't exist or it has no records
        if not self.count():
            self.index(documents, checkpoint=checkpoint)
            return

        # Create transform and stream
        transform = Transform(self, Action.UPSERT, checkpoint=checkpoint)
        stream = Stream(self, Action.UPSERT)

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy") as buffer:
            # Load documents into database and transform to vectors
            ids, _, embeddings = transform(stream(documents), buffer)
            if embeddings is not None:
                # Remove principal components from embeddings, if necessary
                if self.reducer:
                    self.reducer(embeddings)

                # Append embeddings to the index
                self.ann.append(embeddings)

            # Save indexids-ids mapping for indexes with no database
            if ids and not self.database:
                self.ids = self.createids(self.ids + ids)

        # Scoring upsert, if necessary
        # This must occur before graph upsert in order to be available to the graph
        if self.issparse():
            self.scoring.upsert()

        # Subindexes upsert, if necessary
        if self.indexes:
            self.indexes.upsert()

        # Graph upsert, if necessary
        if self.graph:
            self.graph.upsert(Search(self, indexonly=True), Ids(self), self.batchsimilarity)

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
        elif self.ann or self.scoring:
            # Find existing ids
            for uid in ids:
                indices.extend([index for index, value in enumerate(self.ids) if uid == value])

            # Clear embeddings ids
            for index in indices:
                deletes.append(self.ids[index])
                self.ids[index] = None

        # Delete indices for all indexes and data stores
        if indices:
            # Delete ids from ann
            if self.isdense():
                self.ann.delete(indices)

            # Delete ids from scoring
            if self.issparse():
                self.scoring.delete(indices)

            # Delete ids from subindexes
            if self.indexes:
                self.indexes.delete(indices)

            # Delete ids from graph
            if self.graph:
                self.graph.delete(indices)

        return deletes

    def reindex(self, config=None, function=None, **kwargs):
        """
        Recreates embeddings index using config. This method only works if document content storage is enabled.

        Args:
            config: new config
            function: optional function to prepare content for indexing
            kwargs: additional configuration as keyword args
        """

        if self.database:
            # Merge configuration into single dictionary
            config = {**config, **kwargs} if config and kwargs else config if config else kwargs

            # Keep content and objects parameters to ensure database is preserved
            config["content"] = self.config["content"]
            if "objects" in self.config:
                config["objects"] = self.config["objects"]

            # Reset configuration
            self.configure(config)

            # Reset function references
            if self.functions:
                self.functions.reset()

            # Reindex
            if function:
                self.index(function(self.database.reindex(self.config)), True)
            else:
                self.index(self.database.reindex(self.config), True)

    def transform(self, document, category=None, index=None):
        """
        Transforms document into an embeddings vector.

        Args:
            documents: iterable of (id, data, tags), (id, data) or data
            category: category for instruction-based embeddings
            index: index name, if applicable

        Returns:
            embeddings vector
        """

        return self.batchtransform([document], category, index)[0]

    def batchtransform(self, documents, category=None, index=None):
        """
        Transforms documents into embeddings vectors.

        Args:
            documents: iterable of (id, data, tags), (id, data) or data
            category: category for instruction-based embeddings
            index: index name, if applicable

        Returns:
            embeddings vectors
        """

        # Initialize default parameters, if necessary
        self.defaults()

        # Get vector model
        model = self.indexes.model(index) if index and self.indexes else self.model if self.model else self.indexes.model()

        # Convert documents into embeddings
        embeddings = model.batchtransform(Stream(self)(documents), category)

        # Reduce the dimensionality of the embeddings. Scale the embeddings using this
        # model to reduce the noise of common but less relevant terms.
        if self.reducer:
            self.reducer(embeddings)

        return embeddings

    def count(self):
        """
        Total number of elements in this embeddings index.

        Returns:
            number of elements in this embeddings index
        """

        if self.ann:
            return self.ann.count()
        if self.scoring:
            return self.scoring.count()
        if self.database:
            return self.database.count()
        if self.ids:
            return len([uid for uid in self.ids if uid is not None])

        # Default to 0 when no suitable method found
        return 0

    def search(self, query, limit=None, weights=None, index=None, parameters=None, graph=False):
        """
        Finds documents most similar to the input query. This method runs an index search, index + database search
        or a graph search, depending on the embeddings configuration and query.

        Args:
            query: input query
            limit: maximum results
            weights: hybrid score weights, if applicable
            index: index name, if applicable
            parameters: dict of named parameters to bind to placeholders
            graph: return graph results if True

        Returns:
            list of (id, score) for index search
            list of dict for an index + database search
            graph when graph is set to True
        """

        results = self.batchsearch([query], limit, weights, index, [parameters], graph)
        return results[0] if results else results

    def batchsearch(self, queries, limit=None, weights=None, index=None, parameters=None, graph=False):
        """
        Finds documents most similar to the input query. This method runs an index search, index + database search
        or a graph search, depending on the embeddings configuration and query.

        Args:
            queries: input queries
            limit: maximum results
            weights: hybrid score weights, if applicable
            index: index name, if applicable
            parameters: list of dicts of named parameters to bind to placeholders
            graph: return graph results if True

        Returns:
            list of (id, score) per query for index search
            list of dict per query for an index + database search
            list of graph per query when graph is set to True
        """

        # Determine if graphs should be returned
        graph = graph if self.graph else False

        # Execute search
        results = Search(self, indexids=graph)(queries, limit, weights, index, parameters)

        # Create subgraphs using results, if necessary
        return [self.graph.filter(x) if isinstance(x, list) else x for x in results] if graph else results

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
        Explains the importance of each input token in text for a query. This method requires either content to be enabled
        or texts to be provided.

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
        Explains the importance of each input token in text for a list of queries. This method requires either content to be enabled
        or texts to be provided.

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

        # Return true if path has a config.json or config file with an offset set
        return path and (os.path.exists(f"{path}/config.json") or os.path.exists(f"{path}/config")) and "offset" in Configuration().load(path)

    def load(self, path=None, cloud=None, config=None, **kwargs):
        """
        Loads an existing index from path.

        Args:
            path: input path
            cloud: cloud storage configuration
            config: configuration overrides
            kwargs: additional configuration as keyword args

        Returns:
            Embeddings
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
        self.config = Configuration().load(path)

        # Apply config overrides
        self.config = {**self.config, **config} if config else self.config

        # Approximate nearest neighbor index - stores dense vectors
        self.ann = self.createann()
        if self.ann:
            self.ann.load(f"{path}/embeddings")

        # Dimensionality reduction model - word vectors only
        if self.config.get("pca"):
            self.reducer = Reducer()
            self.reducer.load(f"{path}/lsa")

        # Index ids when content is disabled
        self.ids = self.createids()
        if self.ids:
            self.ids.load(f"{path}/ids")

        # Document database - stores document content
        self.database = self.createdatabase()
        if self.database:
            self.database.load(f"{path}/documents")

        # Sparse vectors - stores term sparse arrays
        self.scoring = self.createscoring()
        if self.scoring:
            self.scoring.load(f"{path}/scoring")

        # Subindexes
        self.indexes = self.createindexes()
        if self.indexes:
            self.indexes.load(f"{path}/indexes")

        # Graph network - stores relationships
        self.graph = self.creategraph()
        if self.graph:
            self.graph.load(f"{path}/graph")

        # Dense vectors - transforms data to embeddings vectors
        self.model = self.loadvectors()

        # Query model
        self.query = self.loadquery()

        return self

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

            # Save index configuration
            Configuration().save(self.config, path)

            # Save approximate nearest neighbor index
            if self.ann:
                self.ann.save(f"{path}/embeddings")

            # Save dimensionality reduction model (word vectors only)
            if self.reducer:
                self.reducer.save(f"{path}/lsa")

            # Save index ids
            if self.ids:
                self.ids.save(f"{path}/ids")

            # Save document database
            if self.database:
                self.database.save(f"{path}/documents")

            # Save scoring index
            if self.scoring:
                self.scoring.save(f"{path}/scoring")

            # Save subindexes
            if self.indexes:
                self.indexes.save(f"{path}/indexes")

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

        self.config, self.archive = None, None
        self.reducer, self.query = None, None
        self.ids = None

        # Close ANN
        if self.ann:
            self.ann.close()
            self.ann = None

        # Close database
        if self.database:
            self.database.close()
            self.database, self.functions = None, None

        # Close scoring
        if self.scoring:
            self.scoring.close()
            self.scoring = None

        # Close graph
        if self.graph:
            self.graph.close()
            self.graph = None

        # Close indexes
        if self.indexes:
            self.indexes.close()
            self.indexes = None

        # Close vectors model
        if self.model:
            self.model.close()
            self.model = None

        self.models = None

    def info(self):
        """
        Prints the current embeddings index configuration.
        """

        if self.config:
            # Print configuration
            print(json.dumps(self.config, sort_keys=True, default=str, indent=2))

    def issparse(self):
        """
        Checks if this instance has an associated scoring instance with term indexing enabled.

        Returns:
            True if term index is enabled, False otherwise
        """

        return self.scoring and self.scoring.hasterms()

    def isdense(self):
        """
        Checks if this instance has an associated ANN instance.

        Returns:
            True if this instance has an associated ANN, False otherwise
        """

        return self.ann is not None

    def isweighted(self):
        """
        Checks if this instance has an associated scoring instance with term weighting enabled.

        Returns:
            True if term weighting is enabled, False otherwise
        """

        return self.scoring and not self.scoring.hasterms()

    def configure(self, config):
        """
        Sets the configuration for this embeddings index and loads config-driven models.

        Args:
            config: embeddings configuration
        """

        # Configuration
        self.config = config

        # Dimensionality reduction model
        self.reducer = None

        # Create scoring instance for word vectors term weighting
        scoring = self.config.get("scoring") if self.config else None
        self.scoring = self.createscoring() if scoring and (not isinstance(scoring, dict) or not scoring.get("terms")) else None

        # Dense vectors - transforms data to embeddings vectors
        self.model = self.loadvectors() if self.config else None

        # Query model
        self.query = self.loadquery() if self.config else None

    def initindex(self, reindex):
        """
        Initialize new index.

        Args:
            reindex: if this is a reindex operation in which case database creation is skipped, defaults to False
        """

        # Initialize default parameters, if necessary
        self.defaults()

        # Initialize index ids, only created when content is disabled
        self.ids = None

        # Create document database, if necessary
        if not reindex:
            self.database = self.createdatabase()

            # Reset archive since this is a new index
            self.archive = None

        # Close existing ANN, if necessary
        if self.ann:
            self.ann.close()

        # Initialize ANN, will be created after index transformations complete
        self.ann = None

        # Create scoring only if term indexing is enabled
        scoring = self.config.get("scoring")
        if scoring and isinstance(scoring, dict) and self.config["scoring"].get("terms"):
            self.scoring = self.createscoring()

        # Create subindexes, if necessary
        self.indexes = self.createindexes()

        # Create graph, if necessary
        self.graph = self.creategraph()

    def defaults(self):
        """
        Apply default parameters to current configuration.

        Returns:
            configuration with default parameters set
        """

        self.config = self.config if self.config else {}

        # Expand sparse index shortcuts
        if not self.config.get("scoring") and any(self.config.get(key) for key in ["keyword", "hybrid"]):
            self.config["scoring"] = {"method": "bm25", "terms": True, "normalize": True}

        # Expand graph shortcuts
        if self.config.get("graph") is True:
            self.config["graph"] = {}

        # Check if default model should be loaded
        if not self.model and self.defaultallowed():
            self.config["path"] = "sentence-transformers/all-MiniLM-L6-v2"

            # Load dense vectors model
            self.model = self.loadvectors()

    def defaultallowed(self):
        """
        Tests if this embeddings instance can use a default model if not otherwise provided.

        Returns:
            True if a default model is allowed, False otherwise
        """

        params = [("keyword", False), ("defaults", True)]
        return all(self.config.get(key, default) == default for key, default in params)

    def loadvectors(self):
        """
        Loads a vector model set in config.

        Returns:
            vector model
        """

        # Create model cache if subindexes are enabled
        if "indexes" in self.config and self.models is None:
            self.models = {}

        # Load vector model
        return VectorsFactory.create(self.config, self.scoring, self.models)

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

    def createann(self):
        """
        Creates an ANN from config.

        Returns:
            new ANN, if enabled in config
        """

        # Free existing resources
        if self.ann:
            self.ann.close()

        return ANNFactory.create(self.config) if self.config.get("path") or self.defaultallowed() else None

    def createdatabase(self):
        """
        Creates a database from config. This method will also close any existing database connection.

        Returns:
            new database, if enabled in config
        """

        # Free existing resources
        if self.database:
            self.database.close()

        config = self.config.copy()

        # Create references to callable functions
        self.functions = Functions(self) if "functions" in config else None
        if self.functions:
            config["functions"] = self.functions(config)

        # Create database from config and return
        return DatabaseFactory.create(config)

    def creategraph(self):
        """
        Creates a graph from config.

        Returns:
            new graph, if enabled in config
        """

        # Free existing resources
        if self.graph:
            self.graph.close()

        if "graph" in self.config:
            # Get or create graph configuration
            config = self.config["graph"] if "graph" in self.config else {}

            # Create configuration with custom columns, if necessary
            config = self.columns(config)
            return GraphFactory.create(config)

        return None

    def createids(self, ids=None):
        """
        Creates indexids when content is disabled.

        Args:
            ids: optional ids to add

        Returns:
            new indexids, if content disabled
        """

        # Load index ids when content is disabled
        return IndexIds(self, ids) if not self.config.get("content") else None

    def createindexes(self):
        """
        Creates subindexes from config.

        Returns:
            list of subindexes
        """

        # Free existing resources
        if self.indexes:
            self.indexes.close()

        # Load subindexes
        if "indexes" in self.config:
            indexes = {}
            for index, config in self.config["indexes"].items():
                # Create index with shared model cache
                indexes[index] = Embeddings(config, models=self.models)

            # Wrap as Indexes object
            return Indexes(self, indexes)

        return None

    def createscoring(self):
        """
        Creates a scoring from config.

        Returns:
            new scoring, if enabled in config
        """

        # Free existing resources
        if self.scoring:
            self.scoring.close()

        if "scoring" in self.config:
            # Expand scoring to a dictionary, if necessary
            config = self.config["scoring"]
            config = config if isinstance(config, dict) else {"method": config}

            # Create configuration with custom columns, if necessary
            config = self.columns(config)
            return ScoringFactory.create(config)

        return None

    def columns(self, config):
        """
        Adds custom text/object column information if it's provided.

        Args:
            config: input configuration

        Returns:
            config with column information added
        """

        # Add text/object columns if custom
        if "columns" in self.config:
            # Work on copy of configuration
            config = config.copy()

            # Copy columns to config
            config["columns"] = self.config["columns"]

        return config
