"""
Search module
"""

import numpy as np


class Search:
    """
    Executes a batch search action. A search can be both approximate nearest neighbor and/or database driven.
    """

    def __init__(self, embeddings):
        """
        Creates a new search action.

        Args:
            embeddings: embeddings instance
        """

        self.embeddings = embeddings

        # Alias embeddings attributes
        self.config = embeddings.config
        self.ann = embeddings.ann
        self.database = embeddings.database
        self.transform = embeddings.transform

    def __call__(self, queries, limit):
        """
        Executes a batch search for queries. This method will run either an approximate nearest neighbor (ann) search
        or an approximate nearest neighbor + database search depending on if a database is available.

        Args:
            queries: list of queries
            limit: maximum results

        Returns:
            list of (id, score) per query for ann search, list of dict per query for an ann+database search
        """

        # Return empty results if ANN is not set
        if not self.ann:
            return [[]] * len(queries)

        if self.database:
            return self.dbsearch(queries, limit)

        # Default: execute an approximate nearest neighbor search
        return self.search(queries, limit)

    def search(self, queries, limit):
        """
        Executes an approximate nearest neighbor search.

        Args:
            queries: list of queries
            limit: maximum results

        Returns:
            list of (id, score) per query
        """

        # Convert queries to embedding vectors
        embeddings = np.array([self.transform((None, query, None)) for query in queries])

        # Search approximate nearest neighbor index
        results = self.ann.search(embeddings, limit)

        # Require scores to be greater than 0
        results = [[(i, score) for i, score in r if score > 0] for r in results]

        # Map indexids to ids if "ids" available
        if "ids" in self.config:
            lookup = self.config["ids"]
            return [[(lookup[i], score) for i, score in r] for r in results]

        return results

    def dbsearch(self, queries, limit):
        """
        Executes an approximate nearest neighbor + database search.

        Args:
            queries: list of queries
            limit: maximum results

        Returns:
            list of dict per query
        """

        # Parse queries
        queries = [self.database.parse(query) for query in queries]

        # Override limit with query limit, if applicable
        limit = max(limit, self.limit(queries))

        # Extract embeddings queries as single batch across all queries
        equeries, search, candidates = [], [], 0
        for x, query in enumerate(queries):
            if "similar" in query:
                # Get handle to similar queries
                for params in query["similar"]:
                    # Store the query index and similar query argument (first argument)
                    equeries.append((x, params[0]))

                    # Second argument is similar candidate limit
                    if len(params) > 1 and params[1].isdigit():
                        # Get largest candidate limit across all queries
                        candidates = int(params[1]) if int(params[1]) > candidates else candidates

        # Bulk approximate nearest neighbor search
        if equeries:
            search = self.search([query for _, query in equeries], limit * 10 if not candidates else candidates)

        # Combine approximate nearest neighbor search results with database search results
        results = []
        for x, query in enumerate(queries):
            # Get search indices for this query within bulk query
            indices = [i for i, (y, _) in enumerate(equeries) if x == y]

            # Run the database query
            result = self.database.search(query, [s for i, s in enumerate(search) if i in indices], limit)
            results.append(result)

        return results

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
