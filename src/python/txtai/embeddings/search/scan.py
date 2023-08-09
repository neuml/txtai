"""
Scan module
"""


class Scan:
    """
    Scans indexes for query matches.
    """

    def __init__(self, search, limit, weights, index):
        """
        Creates a new scan instance.

        Args:
            search: index search function
            limit: maximum results
            weights: default hybrid score weights
            index: default index name
        """

        # Index search function
        self.search = search

        # Default query limit
        self.limit = limit

        # Default number of candidates
        self.candidates = None

        # Default query weights
        self.weights = weights

        # Default index
        self.index = index

    def __call__(self, queries):
        """
        Executes a scan for a list of queries.

        Args:
            queries: list of queries to run

        Returns:
            list of (id, score) per query
        """

        # Query results group by unique query clause id
        results = {}

        # Default number of candidates
        default = None

        # Group by index and run
        for index, iqueries in self.parse(queries).items():
            # Query limit to pass to batch search
            candidates = [query.candidates for query in iqueries if query.candidates]
            if not candidates and not default:
                default = self.default(queries)

            candidates = max(candidates) if candidates else default

            # Query weights to pass to batch search
            weights = [query.weights for query in iqueries if query.weights is not None]
            weights = max(weights) if weights else self.weights

            # Index to run query against
            index = index if index else self.index

            # Run index searches
            for x, result in enumerate(self.search([query.text for query in iqueries], candidates, weights, index)):
                # Save query id and results to later join to original query
                results[iqueries[x].uid] = (iqueries[x].qid, result)

        # Sort by query uid and return results
        return [result for _, result in sorted(results.items())]

    def parse(self, queries):
        """
        Parse index query clauses from a list of parsed queries.

        Args:
            queries: list of parsed queries

        Returns:
            index query clauses grouped by index
        """

        results, uid = {}, 0
        for x, query in enumerate(queries):
            if "similar" in query:
                # Extract similar query clauses
                for params in query["similar"]:
                    # Parse query clause parameters
                    clause = Clause(uid, x, params)

                    # Create clause list for index
                    if clause.index not in results:
                        results[clause.index] = []

                    # Add query to index list, increment uid
                    results[clause.index].append(clause)
                    uid += 1

        return results

    def default(self, queries):
        """
        Derives the default number of candidates. The number of candidates are the number of results to bring back
        from index queries. This is an optional argument to similar() clauses.

        For a single query filter clause, the default is the query limit. With multiple filtering clauses, the default is
        10x the query limit. This ensures that limit results are still returned with additional filtering after an index query.

        Args:
            queries: list of queries

        Returns:
            default candidate list size
        """

        multitoken = any(query.get("where") and len(query["where"].split()) > 1 for query in queries)
        return self.limit * 10 if multitoken else self.limit


class Clause:
    """
    Parses and stores query clause parameters.
    """

    def __init__(self, uid, qid, params):
        """
        Creates a new query clause.

        Args:
            uid: query clause id
            qid: query id clause is a part of
            params: query parameters to parse
        """

        self.uid, self.qid = uid, qid
        self.text, self.index = params[0], None
        self.candidates, self.weights = None, None

        # Parse additional similar clause parameters
        if len(params) > 1:
            self.parse(params[1:])

    def parse(self, params):
        """
        Parses clause parameters into this instance.

        Args:
            params: query clause parameters
        """

        for param in params:
            if param.isdigit():
                # Number of query candidates
                self.candidates = int(param)

            elif param.replace(".", "").isdigit():
                # Hybrid score weights
                self.weights = float(param)

            else:
                # Target index
                self.index = param
