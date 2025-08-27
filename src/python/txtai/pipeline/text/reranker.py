"""
Reranker module
"""

from ..base import Pipeline


class Reranker(Pipeline):
    """
    Runs embeddings queries and re-ranks them using a similarity pipeline. Note that content must be enabled with the
    embeddings instance for this to work properly.
    """

    def __init__(self, embeddings, similarity):
        """
        Creates a Reranker pipeline.

        Args:
            embeddings: embeddings instance (content must be enabled)
            similarity: similarity instance
        """

        self.embeddings, self.similarity = embeddings, similarity

    # pylint: disable=W0222
    def __call__(self, query, limit=3, factor=10, **kwargs):
        """
        Runs an embeddings search and re-ranks the results using a Similarity pipeline.

        Args:
            query: query text|list
            limit: maximum results
            factor: factor to multiply limit by for the initial embeddings search
            kwargs: additional arguments to pass to embeddings search

        Returns:
            list of query results rescored using a Similarity pipeline
        """

        queries = [query] if not isinstance(query, list) else query

        # Run searches
        results = self.embeddings.batchsearch(queries, limit * factor, **kwargs)

        # Re-rank using similarity pipeline
        ranked = []
        for x, result in enumerate(results):
            texts = [row["text"] for row in result]

            # Score results and merge
            for uid, score in self.similarity(queries[x], texts):
                result[uid]["score"] = score

            # Sort and take top n sorted results
            ranked.append(sorted(result, key=lambda row: row["score"], reverse=True)[:limit])

        return ranked[0] if isinstance(query, str) else ranked
