"""
Faiss module
"""

import math

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
        # Configure embeddings index. Inner product is equal to cosine similarity on normalized vectors.
        params = self.configure(embeddings.shape[0])
        self.model = index_factory(embeddings.shape[1], params, METRIC_INNER_PRODUCT)

        # Train model
        self.model.train(embeddings)

        # Add embeddings - position in embeddings is used as the id
        self.model.add_with_ids(embeddings, np.arange(embeddings.shape[0], dtype=np.int64))

        # Add id offset and index build metadata
        self.config["offset"] = embeddings.shape[0]
        self.metadata({"components": params})

    def append(self, embeddings):
        new = embeddings.shape[0]

        # Append new ids - position in embeddings + existing offset is used as the id
        self.model.add_with_ids(embeddings, np.arange(self.config["offset"], self.config["offset"] + new, dtype=np.int64))

        # Update id offset and index metadata
        self.config["offset"] += new
        self.metadata(None)

    def delete(self, ids):
        # Remove specified ids
        self.model.remove_ids(np.array(ids, dtype=np.int64))

    def search(self, queries, limit):
        # Run the query
        self.model.nprobe = self.nprobe()
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

    def configure(self, count):
        """
        Configures settings for a new index.

        Args:
            count: initial number of embeddings rows

        Returns:
            user-specified or generated components setting
        """

        # Lookup components setting
        components = self.setting("components")

        if components:
            return components

        # Get storage setting
        storage = "SQ8" if self.config.get("quantize") else "Flat"

        # Small index, use storage directly with IDMap
        if count <= 5000:
            return f"IDMap,{storage}"

        x = self.cells(count)
        components = f"IVF{x},{storage}"

        return components

    def cells(self, count):
        """
        Calculates the number of IVF cells for an IVF index.

        Args:
            count: number of embeddings rows

        Returns:
            number of IVF cells
        """

        # Calculate number of IVF cells where x = min(4 * sqrt(embeddings count), embeddings count / 39)
        # Faiss requires at least 39 * x data points
        return min(round(4 * math.sqrt(count)), int(count / 39))

    def nprobe(self):
        """
        Gets or derives the nprobe search parameter
        """

        # Get size of embeddings index
        count = self.count()

        default = 6 if count <= 5000 else round(self.cells(count) / 16)
        return self.setting("nprobe", default)
