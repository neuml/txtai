"""
Annoy module
"""

# Conditional import
try:
    from annoy import AnnoyIndex

    ANNOY = True
except ImportError:
    ANNOY = False

from .base import ANN


# pylint: disable=W0223
class Annoy(ANN):
    """
    Builds an ANN index using the Annoy library.
    """

    def __init__(self, config):
        super().__init__(config)

        if not ANNOY:
            raise ImportError('Annoy is not available - install "ann" extra to enable')

    def load(self, path):
        # Load index
        self.backend = AnnoyIndex(self.config["dimensions"], self.config["metric"])
        self.backend.load(path)

    def index(self, embeddings):
        # Inner product is equal to cosine similarity on normalized vectors
        self.config["metric"] = "dot"

        # Create index
        self.backend = AnnoyIndex(self.config["dimensions"], self.config["metric"])

        # Add items - position in embeddings is used as the id
        for x in range(embeddings.shape[0]):
            self.backend.add_item(x, embeddings[x])

        # Build index
        ntrees = self.setting("ntrees", 10)
        self.backend.build(ntrees)

        # Add index build metadata
        self.metadata({"ntrees": ntrees})

    def search(self, queries, limit):
        # Lookup search k setting
        searchk = self.setting("searchk", -1)

        # Annoy doesn't have a built in batch query method
        results = []
        for query in queries:
            # Run the query
            ids, scores = self.backend.get_nns_by_vector(query, n=limit, search_k=searchk, include_distances=True)

            # Map results to [(id, score)]
            results.append(list(zip(ids, scores)))

        return results

    def count(self):
        # Number of items in index
        return self.backend.get_n_items()

    def save(self, path):
        # Write index
        self.backend.save(path)
