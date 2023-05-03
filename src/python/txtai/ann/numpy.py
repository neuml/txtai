"""
NumPy module
"""

import pickle

import numpy as np

from .. import __pickle__

from .base import ANN


class NumPy(ANN):
    """
    Builds an ANN index backed by a NumPy array.
    """

    def load(self, path):
        # Load array from file
        with open(path, "rb") as handle:
            self.backend = pickle.load(handle)

    def index(self, embeddings):
        # Create index
        self.backend = embeddings

        # Add id offset and index build metadata
        self.config["offset"] = embeddings.shape[0]

    def append(self, embeddings):
        new = embeddings.shape[0]

        # Append new data to array
        self.backend = np.concatenate((self.backend, embeddings), axis=0)

        # Update id offset
        self.config["offset"] += new

    def delete(self, ids):
        # Filter any index greater than size of array
        ids = [x for x in ids if x < self.backend.shape[0]]

        # Clear specified ids
        self.backend[ids] = np.zeros((len(ids), self.backend.shape[1]))

    def search(self, queries, limit):
        # Dot product on normalized vectors is equal to cosine similarity
        scores = np.dot(queries, self.backend.T).tolist()

        # Add index and sort desc based on score
        return [sorted(enumerate(score), key=lambda x: x[1], reverse=True)[:limit] for score in scores]

    def count(self):
        # Get count of non-zero rows (ignores deleted rows)
        return self.backend[~np.all(self.backend == 0, axis=1)].shape[0]

    def save(self, path):
        # Save array to file
        with open(path, "wb") as handle:
            pickle.dump(self.backend, handle, protocol=__pickle__)
