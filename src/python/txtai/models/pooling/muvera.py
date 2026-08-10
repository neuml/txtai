"""
Muvera module
"""

# Core library imports
from ...util import Library

np = Library().numpy()
torch = Library().torch()


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

    # Signals LatePooling that this encoder must receive true, unpadded token rows
    unpadded = True

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
        lengths = torch.tensor([len(doc) for doc in data], dtype=torch.int64)
        total = lengths.sum().item()
        documents = torch.repeat_interleave(torch.arange(length), lengths)

        # Stack all vectors
        points = torch.cat([torch.as_tensor(doc, dtype=torch.float32) for doc in data])

        # Output vectors
        size = self.repetitions * partitions * projection
        vectors = torch.zeros((length, size), dtype=torch.float32)

        # Process each repetition
        for number in range(self.repetitions):
            seed = self.seed + number

            # Calculate the simhash
            sketches = torch.matmul(points, self.random(dimension, self.hashes, seed))

            # Dimensionality reduction, if necessary
            projected = points if identity else torch.matmul(points, self.reducer(dimension, projection, seed))

            # Get partition indices
            bits = (sketches > 0).to(torch.int64)
            indices = torch.zeros(total, dtype=torch.int64)

            # Calculate vector indices
            for x in range(self.hashes):
                indices = (indices << 1) + (bits[:, x] ^ (indices & 1))

            # Initialize storage
            fdesum = torch.zeros((length * partitions * projection,), dtype=torch.float32)
            counts = torch.zeros((length * partitions,), dtype=torch.int32)

            # Count vectors per partition per document
            flat = documents * partitions + indices
            counts.index_add_(0, flat, torch.ones(total, dtype=torch.int32))
            counts = counts.reshape(length, partitions)

            # Aggregate vectors using flattened indexing for efficiency
            base = flat * projection

            for d in range(projection):
                fdesum.index_add_(0, base + d, projected[:, d])

            # Reshape for easier manipulation
            fdesum = fdesum.reshape(length, partitions, projection)

            # Convert sums to averages for data category
            if category == "data":
                # Safe division (avoid divide by zero)
                counts = counts[:, :, None]
                fdesum = torch.where(counts > 0, fdesum / counts.to(torch.float32), fdesum)

            # Save results
            start = number * partitions * projection
            vectors[:, start : start + partitions * projection] = fdesum.reshape(length, -1)

        return vectors.cpu().numpy()

    def random(self, dimension, projection, seed):
        """
        Generates a random matrix for simhash projections.

        Args:
            dimension: number of dimensions for input vectors
            projection: number of projection dimensions
            seed: random seed

        Returns:
            random matrix for simhash projections
        """

        rng = np.random.default_rng(seed)
        return torch.from_numpy(rng.normal(loc=0.0, scale=1.0, size=(dimension, projection)).astype(np.float32))

    def reducer(self, dimension, projection, seed):
        """
        Generates a random matrix for dimensionality reduction using the AMS sketch algorithm.

        Args:
            dimension: number of input dimensions
            projection: number of dimensions to project inputs to

        Returns:
            Dimensionality reduced matrix
        """

        rng = np.random.default_rng(seed)
        out = np.zeros((dimension, projection), dtype=np.float32)
        indices = rng.integers(0, projection, size=dimension)
        signs = rng.choice([-1.0, 1.0], size=dimension)
        out[np.arange(dimension), indices] = signs

        return torch.from_numpy(out)
