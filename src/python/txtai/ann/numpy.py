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

    def __init__(self, config):
        super().__init__(config)

        # Array function definitions
        self.all, self.cat, self.dot, self.zeros = np.all, np.concatenate, np.dot, np.zeros

    def load(self, path):
        # Load array from file
        with open(path, "rb") as handle:
            self.backend = self.tensor(pickle.load(handle))

    def index(self, embeddings):
        # Create index
        self.backend = self.tensor(embeddings)

        # Add id offset and index build metadata
        self.config["offset"] = embeddings.shape[0]
        self.metadata(self.settings())

    def append(self, embeddings):
        new = embeddings.shape[0]

        # Append new data to array
        self.backend = self.cat((self.backend, self.tensor(embeddings)), axis=0)

        # Update id offset and index metadata
        self.config["offset"] += new
        self.metadata()

    def delete(self, ids):
        # Filter any index greater than size of array
        ids = [x for x in ids if x < self.backend.shape[0]]

        # Clear specified ids
        self.backend[ids] = self.tensor(self.zeros((len(ids), self.backend.shape[1])))

    def search(self, queries, limit):
        # Dot product on normalized vectors is equal to cosine similarity
        scores = self.dot(self.tensor(queries), self.backend.T).tolist()

        # Add index and sort desc based on score
        return [sorted(enumerate(score), key=lambda x: x[1], reverse=True)[:limit] for score in scores]

    def count(self):
        # Get count of non-zero rows (ignores deleted rows)
        return self.backend[~self.all(self.backend == 0, axis=1)].shape[0]

    def save(self, path):
        # Save array to file
        with open(path, "wb") as handle:
            pickle.dump(self.backend, handle, protocol=__pickle__)

    def tensor(self, array):
        """
        Handles backend-specific code such as loading to a GPU device.

        Args:
            array: data array

        Returns:
            array with backend-specific logic applied
        """

        return array

    def settings(self):
        """
        Returns settings for this array.

        Returns:
            dict
        """

        return {"numpy": np.__version__}
