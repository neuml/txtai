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
    Builds an ANN model using the Annoy library.
    """

    def load(self, path):
        # Load index
        self.model = AnnoyIndex(self.config["dimensions"], self.config["metric"])
        self.model.load(path)

    def index(self, embeddings):
        # Inner product is equal to cosine similarity on normalized vectors
        self.config["metric"] = "dot"

        # Create index
        self.model = AnnoyIndex(self.config["dimensions"], self.config["metric"])

        # Add items - position in embeddings is used as the id
        for x in range(embeddings.shape[0]):
            self.model.add_item(x, embeddings[x])

        # Build index
        ntrees = self.setting("ntrees", 10)
        self.model.build(ntrees)

        # Add index build metadata
        self.metadata({"ntrees": ntrees})

    def search(self, queries, limit):
        # Lookup search k setting
        searchk = self.setting("searchk", -1)

        # Annoy doesn't have a built in batch query method
        results = []
        for query in queries:
            # Run the query
            ids, scores = self.model.get_nns_by_vector(query, n=limit, search_k=searchk, include_distances=True)

            # Map results to [(id, score)]
            results.append(list(zip(ids, scores)))

        return results

    def count(self):
        # Number of items in index
        return self.model.get_n_items()

    def save(self, path):
        # Write index
        self.model.save(path)
