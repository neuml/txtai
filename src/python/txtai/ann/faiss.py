"""
Faiss module
"""

import math
import platform

import numpy as np

from faiss import omp_set_num_threads
from faiss import index_factory, IO_FLAG_MMAP, METRIC_INNER_PRODUCT, read_index, write_index
from faiss import index_binary_factory, read_index_binary, write_index_binary, IndexBinaryIDMap

from .base import ANN

if platform.system() == "Darwin":
    # Workaround for a Faiss issue causing segmentation faults on macOS. See txtai FAQ for more.
    omp_set_num_threads(1)


class Faiss(ANN):
    """
    Builds an ANN index using the Faiss library.
    """

    def __init__(self, config):
        super().__init__(config)

        # Scalar quantization
        quantize = self.config.get("quantize")
        self.qbits = quantize if quantize and isinstance(quantize, int) and not isinstance(quantize, bool) else None

    def load(self, path):
        # Get read function
        readindex = read_index_binary if self.qbits else read_index

        # Load index
        self.backend = readindex(path, IO_FLAG_MMAP if self.setting("mmap") is True else 0)

    def index(self, embeddings):
        # Compute model training size
        train, sample = embeddings, self.setting("sample")
        if sample:
            # Get sample for training
            rng = np.random.default_rng(0)
            indices = sorted(rng.choice(train.shape[0], int(sample * train.shape[0]), replace=False, shuffle=False))
            train = train[indices]

        # Configure embeddings index. Inner product is equal to cosine similarity on normalized vectors.
        params = self.configure(embeddings.shape[0], train.shape[0])

        # Create index
        self.backend = self.create(embeddings, params)

        # Train model
        self.backend.train(train)

        # Add embeddings - position in embeddings is used as the id
        self.backend.add_with_ids(embeddings, np.arange(embeddings.shape[0], dtype=np.int64))

        # Add id offset and index build metadata
        self.config["offset"] = embeddings.shape[0]
        self.metadata({"components": params})

    def append(self, embeddings):
        new = embeddings.shape[0]

        # Append new ids - position in embeddings + existing offset is used as the id
        self.backend.add_with_ids(embeddings, np.arange(self.config["offset"], self.config["offset"] + new, dtype=np.int64))

        # Update id offset and index metadata
        self.config["offset"] += new
        self.metadata()

    def delete(self, ids):
        # Remove specified ids
        self.backend.remove_ids(np.array(ids, dtype=np.int64))

    def search(self, queries, limit):
        # Set nprobe and nflip search parameters
        self.backend.nprobe = self.nprobe()
        self.backend.nflip = self.setting("nflip", self.backend.nprobe)

        # Run the query
        scores, ids = self.backend.search(queries, limit)

        # Map results to [(id, score)]
        results = []
        for x, score in enumerate(scores):
            # Transform scores and add results
            results.append(list(zip(ids[x].tolist(), self.scores(score))))

        return results

    def count(self):
        return self.backend.ntotal

    def save(self, path):
        # Get write function
        writeindex = write_index_binary if self.qbits else write_index

        # Write index
        writeindex(self.backend, path)

    def configure(self, count, train):
        """
        Configures settings for a new index.

        Args:
            count: initial number of embeddings rows
            train: number of rows selected for model training

        Returns:
            user-specified or generated components setting
        """

        # Lookup components setting
        components = self.setting("components")

        if components:
            # Format and return components string
            return self.components(components, train)

        # Derive quantization. Prefer backend-specific setting. Fallback to root-level parameter.
        quantize = self.setting("quantize", self.config.get("quantize"))
        quantize = 8 if isinstance(quantize, bool) else quantize

        # Get storage setting
        storage = f"SQ{quantize}" if quantize else "Flat"

        # Small index, use storage directly with IDMap
        if count <= 5000:
            return "BFlat" if self.qbits else f"IDMap,{storage}"

        x = self.cells(train)
        components = f"BIVF{x}" if self.qbits else f"IVF{x},{storage}"

        return components

    def create(self, embeddings, params):
        """
        Creates a new index.

        Args:
            embeddings: embeddings to index
            params: index parameters

        Returns:
            new index
        """

        # Create binary index
        if self.qbits:
            index = index_binary_factory(embeddings.shape[1] * 8, params)

            # Wrap with BinaryIDMap, if necessary
            if any(x in params for x in ["BFlat", "BHNSW"]):
                index = IndexBinaryIDMap(index)

            return index

        # Create standard float index
        return index_factory(embeddings.shape[1], params, METRIC_INNER_PRODUCT)

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
        return max(min(round(4 * math.sqrt(count)), int(count / 39)), 1)

    def components(self, components, train):
        """
        Formats a components string. This method automatically calculates the optimal number of IVF cells, if omitted.

        Args:
            components: input components string
            train: number of rows selected for model training

        Returns:
            formatted components string
        """

        # Optimal number of IVF cells
        x = self.cells(train)

        # Add number of IVF cells, if missing
        components = [f"IVF{x}" if component == "IVF" else component for component in components.split(",")]

        # Return components string
        return ",".join(components)

    def nprobe(self):
        """
        Gets or derives the nprobe search parameter.

        Returns:
            nprobe setting
        """

        # Get size of embeddings index
        count = self.count()

        default = 6 if count <= 5000 else round(self.cells(count) / 16)
        return self.setting("nprobe", default)

    def scores(self, scores):
        """
        Calculates the index score from the input score. This method returns the hamming score
        (1.0 - (hamming distance / total number of bits)) for binary indexes and the input
        scores otherwise.

        Args:
            scores: input scores

        Returns:
            index scores
        """

        # Calculate hamming score, bound between 0.0 - 1.0
        if self.qbits:
            return np.clip(1.0 - (scores / (self.config["dimensions"] * 8)), 0.0, 1.0).tolist()

        # Standard scoring
        return scores.tolist()
