"""
HNSW module
"""

import numpy as np

# Conditional import
try:
    # pylint: disable=E0611
    from hnswlib import Index

    HNSWLIB = True
except ImportError:
    HNSWLIB = False

from .base import ANN


class HNSW(ANN):
    """
    Builds an ANN model using the hnswlib library.
    """

    def load(self, path):
        # Load index
        self.model = Index(dim=self.config["dimensions"], space=self.config["metric"])
        self.model.load_index(path)

    def index(self, embeddings):
        # Inner product is equal to cosine similarity on normalized vectors
        self.config["metric"] = "ip"

        # Lookup index settings
        efconstruction = self.setting("efconstruction", 200)
        m = self.setting("m", 16)
        seed = self.setting("randomseed", 100)

        # Create index
        self.model = Index(dim=self.config["dimensions"], space=self.config["metric"])
        self.model.init_index(max_elements=embeddings.shape[0], ef_construction=efconstruction, M=m, random_seed=seed)

        # Add items - position in embeddings is used as the id
        self.model.add_items(embeddings, np.arange(embeddings.shape[0]))

        # Add id offset, delete counter and index build metadata
        self.config["offset"] = embeddings.shape[0]
        self.config["deletes"] = 0
        self.metadata({"efconstruction": efconstruction, "m": m, "seed": seed})

    def append(self, embeddings):
        new = embeddings.shape[0]

        # Resize index
        self.model.resize_index(self.config["offset"] + new)

        # Append new ids - position in embeddings + existing offset is used as the id
        self.model.add_items(embeddings, np.arange(self.config["offset"], self.config["offset"] + new))

        # Update id offset and index metadata
        self.config["offset"] += new
        self.metadata(None)

    def delete(self, ids):
        # Mark elements as deleted to omit from search results
        for uid in ids:
            try:
                self.model.mark_deleted(uid)
                self.config["deletes"] += 1
            except RuntimeError:
                # Ignore label not found error
                continue

    def search(self, queries, limit):
        # Set ef query param
        ef = self.setting("efsearch")
        if ef:
            self.model.set_ef(ef)

        # Run the query
        ids, distances = self.model.knn_query(queries, k=limit)

        # Map results to [(id, score)]
        results = []
        for x, distance in enumerate(distances):
            # Convert distances to similarity scores
            scores = [1 - d for d in distance]

            # Build (id, score) tuples, convert np.int64 to python int
            results.append(list(zip(ids[x].tolist(), scores)))

        return results

    def count(self):
        return self.model.get_current_count() - self.config["deletes"]

    def save(self, path):
        # Write index
        self.model.save_index(path)
