"""
Faiss module
"""

import numpy as np

from faiss import index_factory, METRIC_INNER_PRODUCT, read_index, write_index

from .base import ANN


class Faiss(ANN):
    """
    Builds an ANN model using the Faiss library.
    """

    def load(self, path):
        # Load index
        self.model = read_index(path)

    def index(self, embeddings):
        # Lookup components setting
        components = self.setting("components")

        # Create embeddings index. Inner product is equal to cosine similarity on normalized vectors.
        if components:
            params = components
        elif self.config.get("quantize"):
            params = "IVF100,SQ8" if embeddings.shape[0] >= 5000 else "IDMap,SQ8"
        else:
            params = "IVF100,Flat" if embeddings.shape[0] >= 5000 else "IDMap,Flat"

        self.model = index_factory(embeddings.shape[1], params, METRIC_INNER_PRODUCT)

        # Train model
        self.model.train(embeddings)
        self.model.add_with_ids(embeddings, np.arange(embeddings.shape[0], dtype=np.int64))

        # Update id offset
        self.config["offset"] = embeddings.shape[0]

    def append(self, embeddings):
        new = embeddings.shape[0]

        # Append new ids
        self.model.add_with_ids(embeddings, np.arange(self.config["offset"], self.config["offset"] + new, dtype=np.int64))

        # Update id offset
        self.config["offset"] += new

    def delete(self, ids):
        # Remove specified ids
        self.model.remove_ids(np.array(ids, dtype=np.int64))

    def search(self, queries, limit):
        # Run the query
        self.model.nprobe = self.setting("nprobe", 6)
        scores, ids = self.model.search(queries, limit)

        # Map results to [(id, score)]
        results = []
        for x, score in enumerate(scores):
            results.append(list(zip(ids[x].tolist(), score.tolist())))

        return results

    def count(self):
        return self.model.ntotal

    def save(self, path):
        # Write index
        write_index(self.model, path)
