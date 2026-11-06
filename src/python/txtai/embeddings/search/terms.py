"""
Terms module
"""


class Terms:
    """
    Reduces a query statement down to keyword terms. This method extracts the query text from similar clauses if it's a SQL statement.
    Otherwise, the original query is returned.
    """

    def __init__(self, embeddings):
        """
        Create a new terms action.

        Args:
            embeddings: embeddings instance
        """

        self.database = embeddings.database

    def __call__(self, queries):
        """
        Extracts keyword terms from a list of queries.

        Args:
            queries: list of queries

        Returns:
            list of queries reduced down to keyword term strings
        """

        # Parse queries and extract keyword terms for each query
        if self.database:
            terms = []
            for query in queries:
                # Parse query
                parse = self.database.parse(query)

                # Join query text from similar clauses. Only the first argument is query text, the rest are
                # candidates, index and weights. A query without a similar() clause has no keyword terms,
                # add None to keep the output aligned with the input queries.
                terms.append(" ".join(s[0] for s in parse["similar"]) if "similar" in parse else None)

            return terms

        # Return original query when database is None
        return queries
