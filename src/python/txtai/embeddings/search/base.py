"""
Search module
"""

import logging

from .errors import IndexNotFoundError
from .scan import Scan

# Logging configuration
logger = logging.getLogger(__name__)


class Search:
    """
    Executes a batch search action. A search can be both index and/or database driven.
    """

    def __init__(self, embeddings, indexids=False, indexonly=False):
        """
        Creates a new search action.

        Args:
            embeddings: embeddings instance
            indexids: searches return indexids when True, otherwise run standard search
            indexonly: always runs an index search even when a database is available
        """

        self.embeddings = embeddings
        self.indexids = indexids or indexonly
        self.indexonly = indexonly

        # Alias embeddings attributes
        self.ann = embeddings.ann
        self.batchtransform = embeddings.batchtransform
        self.database = embeddings.database
        self.ids = embeddings.ids
        self.indexes = embeddings.indexes
        self.graph = embeddings.graph
        self.query = embeddings.query
        self.scoring = embeddings.scoring if embeddings.issparse() else None

    def __call__(self, queries, limit=None, weights=None, index=None, parameters=None):
        """
        Executes a batch search for queries. This method will run either an index search or an index + database search
        depending on if a database is available.

        Args:
            queries: list of queries
            limit: maximum results
            weights: hybrid score weights
            index: index name
            parameters: list of dicts of named parameters to bind to placeholders

        Returns:
            list of (id, score) per query for index search
            list of dict per query for an index + database search
            list of graph results for a graph index search
        """

        # Default input parameters
        limit = limit if limit else 3
        weights = weights if weights is not None else 0.5

        # Return empty results if there is no database and indexes
        if not self.ann and not self.scoring and not self.indexes and not self.database:
            return [[]] * len(queries)

        # Default index name if only subindexes set
        if not index and not self.ann and not self.scoring and self.indexes:
            index = self.indexes.default()

        # Graph search
        if self.graph and self.graph.isquery(queries):
            return self.graphsearch(queries, limit, weights, index)

        # Database search
        if not self.indexonly and self.database:
            return self.dbsearch(queries, limit, weights, index, parameters)

        # Default vector index query (sparse, dense or hybrid)
        return self.search(queries, limit, weights, index)

    def search(self, queries, limit, weights, index):
        """
        Executes an index search. When only a sparse index is enabled, this is a a keyword search. When only
        a dense index is enabled, this is an ann search. When both are enabled, this is a hybrid search.

        This method will also query subindexes, if available.

        Args:
            queries: list of queries
            limit: maximum results
            weights: hybrid score weights
            index: index name

        Returns:
            list of (id, score) per query
        """

        # Run against specified subindex
        if index:
            return self.subindex(queries, limit, weights, index)

        # Run against base index
        dense = self.dense(queries, limit) if self.ann else None
        sparse = self.sparse(queries, limit) if self.scoring else None

        # Combine scores together
        if dense and sparse:
            # Create weights array if single number passed
            if isinstance(weights, (int, float)):
                weights = [weights, 1 - weights]

            # Create weighted scores
            results = []
            for vectors in zip(dense, sparse):
                uids = {}
                for v, scores in enumerate(vectors):
                    for r, (uid, score) in enumerate(scores if weights[v] > 0 else []):
                        # Initialize score
                        if uid not in uids:
                            uids[uid] = 0.0

                        # Create hybrid score
                        #  - Convex Combination when sparse scores are normalized
                        #  - Reciprocal Rank Fusion (RRF) when sparse scores aren't normalized
                        if self.scoring.isnormalized():
                            uids[uid] += score * weights[v]
                        else:
                            uids[uid] += (1.0 / (r + 1)) * weights[v]

                results.append(sorted(uids.items(), key=lambda x: x[1], reverse=True)[:limit])

            return results

        # Raise an error if when no indexes are available
        if not sparse and not dense:
            raise IndexNotFoundError("No indexes available")

        # Return single query results
        return dense if dense else sparse

    def subindex(self, queries, limit, weights, index):
        """
        Executes a subindex search.

        Args:
            queries: list of queries
            limit: maximum results
            weights: hybrid score weights
            index: index name

        Returns:
            list of (id, score) per query
        """

        # Check that index exists
        if not self.indexes or index not in self.indexes:
            raise IndexNotFoundError(f"Index '{index}' not found")

        # Run subindex search
        results = self.indexes[index].batchsearch(queries, limit, weights)
        return self.resolve(results)

    def dense(self, queries, limit):
        """
        Executes an dense vector search with an approximate nearest neighbor index.

        Args:
            queries: list of queries
            limit: maximum results

        Returns:
            list of (id, score) per query
        """

        # Convert queries to embedding vectors
        embeddings = self.batchtransform((None, query, None) for query in queries)

        # Search approximate nearest neighbor index
        results = self.ann.search(embeddings, limit)

        # Require scores to be greater than 0
        results = [[(i, score) for i, score in r if score > 0] for r in results]

        return self.resolve(results)

    def sparse(self, queries, limit):
        """
        Executes a sparse vector search with a term frequency sparse array.

        Args:
            queries: list of queries
            limit: maximum results

        Returns:
            list of (id, score) per query
        """

        # Search term frequency sparse index
        return self.resolve(self.scoring.batchsearch(queries, limit))

    def resolve(self, results):
        """
        Resolves index ids. This is only executed when content is disabled.

        Args:
            results: results

        Returns:
            results with resolved ids
        """

        # Map indexids to ids if embeddings ids are available
        if not self.indexids and self.ids:
            return [[(self.ids[i], score) for i, score in r] for r in results]

        return results

    def dbsearch(self, queries, limit, weights, index, parameters):
        """
        Executes an index + database search.

        Args:
            queries: list of queries
            limit: maximum results
            weights: default hybrid score weights
            index: default index name
            parameters: list of dicts of named parameters to bind to placeholders

        Returns:
            list of dict per query
        """

        # Parse queries
        queries = self.parse(queries)

        # Override limit with query limit, if applicable
        limit = max(limit, self.limit(queries))

        # Bulk index scan
        scan = Scan(self.search, limit, weights, index)(queries, parameters)

        # Combine index search results with database search results
        results = []
        for x, query in enumerate(queries):
            # Run the database query, get matching bulk searches for current query
            result = self.database.search(
                query, [r for y, r in scan if x == y], limit, parameters[x] if parameters and parameters[x] else None, self.indexids
            )
            results.append(result)

        return results

    def parse(self, queries):
        """
        Parses a list of database queries.

        Args:
            queries: list of queries

        Returns:
            parsed queries
        """

        # Parsed queries
        parsed = []

        for query in queries:
            # Parse query
            parse = self.database.parse(query)

            # Transform query if SQL not parsed and reparse
            if self.query and "select" not in parse:
                # Generate query
                query = self.query(query)
                logger.debug(query)

                # Reparse query
                parse = self.database.parse(query)

            parsed.append(parse)

        return parsed

    def limit(self, queries):
        """
        Parses the largest LIMIT clause from queries.

        Args:
            queries: list of queries

        Returns:
            largest limit number or 0 if not found
        """

        # Override limit with largest limit from database queries
        qlimit = 0
        for query in queries:
            # Parse out qlimit
            l = query.get("limit")
            if l and l.isdigit():
                l = int(l)

            qlimit = l if l and l > qlimit else qlimit

        return qlimit

    def graphsearch(self, queries, limit, weights, index):
        """
        Executes an index + graph search.

        Args:
            queries: list of queries
            limit: maximum results
            weights: default hybrid score weights
            index: default index name

        Returns:
            graph search results
        """

        # Parse queries
        queries = [self.graph.parse(query) for query in queries]

        # Override limit with query limit, if applicable
        limit = max(limit, self.limit(queries))

        # Bulk index scan
        scan = Scan(self.search, limit, weights, index)(queries, None)

        # Combine index search results with database search results
        for x, query in enumerate(queries):
            # Add search results to query
            query["results"] = [r for y, r in scan if x == y]

        return self.graph.batchsearch(queries, limit, self.indexids)
