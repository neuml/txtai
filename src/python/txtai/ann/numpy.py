"""
NumPy module
"""

import numpy as np

from ..serialize import SerializeFactory

from .base import ANN


class NumPy(ANN):
    """
    Builds an ANN index backed by a NumPy array.
    """

    def __init__(self, config):
        super().__init__(config)

        # Array function definitions
        self.all, self.cat, self.dot, self.zeros = np.all, np.concatenate, np.dot, np.zeros
        self.argsort, self.xor, self.clip = np.argsort, np.bitwise_xor, np.clip

        # Scalar quantization
        quantize = self.config.get("quantize")
        self.qbits = quantize if quantize and isinstance(quantize, int) and not isinstance(quantize, bool) else None

    def load(self, path):
        # Load array from file
        try:
            self.backend = self.tensor(np.load(path, allow_pickle=False))
        except ValueError:
            # Backwards compatible support for previously pickled data
            self.backend = self.tensor(SerializeFactory.create("pickle").load(path))

    def index(self, embeddings):
        # Create index
        self.backend = self.tensor(embeddings)

        # Add id offset and index build metadata
        self.config["offset"] = embeddings.shape[0]
        self.metadata(self.settings())

    def append(self, embeddings):
        # Append new data to array
        self.backend = self.cat((self.backend, self.tensor(embeddings)), axis=0)

        # Update id offset and index metadata
        self.config["offset"] += embeddings.shape[0]
        self.metadata()

    def delete(self, ids):
        # Filter any index greater than size of array
        ids = [x for x in ids if x < self.backend.shape[0]]

        # Clear specified ids
        self.backend[ids] = self.tensor(self.zeros((len(ids), self.backend.shape[1])))

    def search(self, queries, limit):
        if self.qbits:
            # Calculate hamming score for integer vectors
            scores = self.hammingscore(queries)
        else:
            # Dot product on normalized vectors is equal to cosine similarity
            scores = self.dot(self.tensor(queries), self.backend.T)

        # Get topn ids
        ids = self.argsort(-scores)[:, :limit]

        # Map results to [(id, score)]
        results = []
        for x, score in enumerate(scores):
            # Add results
            results.append(list(zip(ids[x].tolist(), score[ids[x]].tolist())))

        return results

    def count(self):
        # Get count of non-zero rows (ignores deleted rows)
        return self.backend[~self.all(self.backend == 0, axis=1)].shape[0]

    def save(self, path):
        # Save array to file. Use stream to prevent ".npy" suffix being added.
        with open(path, "wb") as handle:
            np.save(handle, self.numpy(self.backend), allow_pickle=False)

    def tensor(self, array):
        """
        Handles backend-specific code such as loading to a GPU device.

        Args:
            array: data array

        Returns:
            array with backend-specific logic applied
        """

        return array

    def numpy(self, array):
        """
        Handles backend-specific code to convert an array to numpy

        Args:
            array: data array

        Returns:
            numpy array
        """

        return array

    def totype(self, array, dtype):
        """
        Casts array to dtype.

        Args:
            array: input array
            dtype: dtype

        Returns:
            array cast as dtype
        """

        return np.int64(array) if dtype == np.int64 else array

    def settings(self):
        """
        Returns settings for this array.

        Returns:
            dict
        """

        return {"numpy": np.__version__}

    def hammingscore(self, queries):
        """
        Calculates a hamming distance score.

        This is defined as:

            score = 1.0 - (hamming distance / total number of bits)

        Args:
            queries: queries array

        Returns:
            scores
        """

        # Build table of number of bits for each distinct uint8 value
        table = 1 << np.arange(8)
        table = self.tensor(np.array([np.count_nonzero(x & table) for x in np.arange(256)]))

        # Number of different bits
        delta = self.xor(self.tensor(queries[:, None]), self.backend)

        # Cast to long array
        delta = self.totype(delta, np.int64)

        # Calculate score as 1.0 - percentage of different bits
        # Bound score from 0 to 1
        return self.clip(1.0 - (table[delta].sum(axis=2) / (self.config["dimensions"] * 8)), 0.0, 1.0)
