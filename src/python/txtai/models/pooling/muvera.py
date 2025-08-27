"""
Muvera module
"""

import numpy as np


class Muvera:
    """
    Implements the MUVERA (Multi-Vector Retrieval via Fixed Dimensional Encodings) algorithm. This reduces late interaction multi-vector
    outputs to a single fixed vector.

    The size of the output vectors are set using the following parameters

        output dimensions = repetitions * 2^hashes * projected

    For example, the default parameters create vectors with the following output dimensions.

        output dimensions = 20 * 2^5 * 16 = 10240

    This code is based on the following:
      - Paper: https://arxiv.org/abs/2405.19504
      - GitHub: https://github.com/google/graph-mining/tree/main/sketching/point_cloud
      - Python port of the original C++ code: https://github.com/sigridjineth/muvera-py
    """

    def __init__(self, repetitions=20, hashes=5, projection=16, seed=42):
        """
        Creates a Muvera instance.

        Args:
            repetitions: number of iterations
            hashes: number of simhash partitions as 2^hashes
            projection: dimensionality reduction, uses an identity projection when set to None
            seed: random seed
        """

        # Number of repetitions
        self.repetitions = repetitions

        # Number of simhash projections
        self.hashes = hashes

        # Optional number of projected dimensions
        self.projection = projection

        # Seed
        self.seed = seed

    def __call__(self, data, category):
        """
        Transforms a list of multi-vector collections into single fixed vector outputs.

        Args:
            data: array of multi-vector vectors
            category: embeddings category (query or data)
        """

        # Get stats
        dimension, length = data[0].shape[1], len(data)

        # Determine projection dimension
        identity = not self.projection
        projection = dimension if identity else self.projection

        # Number of simhash partitions
        partitions = 2**self.hashes

        # Document tracking
        lengths = np.array([len(doc) for doc in data], dtype=np.int32)
        total = np.sum(lengths)
        documents = np.repeat(np.arange(length), lengths)

        # Stack all vectors
        points = np.vstack(data).astype(np.float32)

        # Output vectors
        size = self.repetitions * partitions * projection
        vectors = np.zeros((length, size), dtype=np.float32)

        # Process each repetition
        for number in range(self.repetitions):
            seed = self.seed + number

            # Calculate the simhash
            sketches = points @ self.random(dimension, self.hashes, seed)

            # Dimensionality reduction, if necessary
            projected = points if identity else (points @ self.reducer(dimension, projection, seed))

            # Get partition indices
            bits = (sketches > 0).astype(np.uint32)
            indices = np.zeros(total, dtype=np.uint32)

            # Calculate vector indices
            for x in range(self.hashes):
                indices = (indices << 1) + (bits[:, x] ^ (indices & 1))

            # Initialize storage
            fdesum = np.zeros((length * partitions * projection,), dtype=np.float32)
            counts = np.zeros((length, partitions), dtype=np.int32)

            # Count vectors per partition per document
            np.add.at(counts, (documents, indices), 1)

            # Aggregate vectors using flattened indexing for efficiency
            part = documents * partitions + indices
            base = part * projection

            for d in range(projection):
                flat = base + d
                np.add.at(fdesum, flat, projected[:, d])

            # Reshape for easier manipulation
            # pylint: disable=E1121
            fdesum = fdesum.reshape(length, partitions, projection)

            # Convert sums to averages for data category
            if category == "data":
                # Safe division (avoid divide by zero)
                counts = counts[:, :, np.newaxis]
                np.divide(fdesum, counts, out=fdesum, where=counts > 0)

            # Save results
            start = number * partitions * projection
            vectors[:, start : start + partitions * projection] = fdesum.reshape(length, -1)

        return vectors

    def random(self, dimension, projection, seed):
        """
        Generates a random matrix for simhash projections.

        Args:
            dimensions: number of dimensions for input vectors
            projections: number of projection dimensions
            seed: random seed

        Returns:
            random matrix for simhash projections
        """

        rng = np.random.default_rng(seed)
        return rng.normal(loc=0.0, scale=1.0, size=(dimension, projection)).astype(np.float32)

    def reducer(self, dimension, projection, seed):
        """
        Generates a random matrix for dimensionality reduction using the AMS sketch algorithm.

        Args:
            dimension: number of input dimensions
            projected: number of dimensions to project inputs to

        Returns:
            Dimensionality reduced matrix
        """

        rng = np.random.default_rng(seed)
        out = np.zeros((dimension, projection), dtype=np.float32)
        indices = rng.integers(0, projection, size=dimension)
        signs = rng.choice([-1.0, 1.0], size=dimension)
        out[np.arange(dimension), indices] = signs

        return out
