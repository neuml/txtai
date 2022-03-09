"""
API module
"""

from .cluster import Cluster

from ..app import Application


class API(Application):
    """
    Base API template. The API is an extended txtai application, adding the ability to cluster API instances together.

    Downstream applications can extend this base template to add/modify functionality.
    """

    def __init__(self, config, loaddata=True):
        super().__init__(config, loaddata)

        # Embeddings cluster
        self.cluster = None
        if self.config.get("cluster"):
            self.cluster = Cluster(self.config["cluster"])

    # pylint: disable=W0221
    def search(self, query, request=None, limit=None):
        # When search is invoked via the API, limit is set from the request
        # When search is invoked directly, limit is set using the method parameter
        limit = self.limit(request.query_params.get("limit") if request and hasattr(request, "query_params") else limit)

        if self.cluster:
            return self.cluster.search(query, limit)

        return super().search(query, limit)

    def batchsearch(self, queries, limit=None):
        if self.cluster:
            return self.cluster.batchsearch(queries, self.limit(limit))

        return super().batchsearch(queries, limit)

    def add(self, documents):
        """
        Adds a batch of documents for indexing.

        Downstream applications can override this method to also store full documents in an external system.

        Args:
            documents: list of {id: value, text: value}

        Returns:
            unmodified input documents
        """

        if self.cluster:
            self.cluster.add(documents)
        else:
            super().add(documents)

        return documents

    def index(self):
        """
        Builds an embeddings index for previously batched documents.
        """

        if self.cluster:
            self.cluster.index()
        else:
            super().index()

    def upsert(self):
        """
        Runs an embeddings upsert operation for previously batched documents.
        """

        if self.cluster:
            self.cluster.upsert()
        else:
            super().upsert()

    def delete(self, ids):
        """
        Deletes from an embeddings index. Returns list of ids deleted.

        Args:
            ids: list of ids to delete

        Returns:
            ids deleted
        """

        if self.cluster:
            return self.cluster.delete(ids)

        return super().delete(ids)

    def count(self):
        """
        Total number of elements in this embeddings index.

        Returns:
            number of elements in embeddings index
        """

        if self.cluster:
            return self.cluster.count()

        return super().count()

    def limit(self, limit):
        """
        Parses the number of results to return from the request. Allows range of 1-250, with a default of 10.

        Args:
            limit: limit parameter

        Returns:
            bounded limit
        """

        # Return between 1 and 250 results, defaults to 10
        return max(1, min(250, int(limit) if limit else 10))
