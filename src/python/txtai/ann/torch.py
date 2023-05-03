"""
PyTorch module
"""

import pickle

import numpy as np
import torch

from .. import __pickle__

from .base import ANN


class Torch(ANN):
    """
    Builds an ANN index backed by a PyTorch array.
    """

    def load(self, path):
        # Load array from file
        with open(path, "rb") as handle:
            self.backend = self.tensor(pickle.load(handle))

    def index(self, embeddings):
        # Create index
        self.backend = self.tensor(embeddings)

        # Add id offset and index build metadata
        self.config["offset"] = embeddings.shape[0]

    def append(self, embeddings):
        new = embeddings.shape[0]

        # Append new data to array
        self.backend = torch.cat((self.backend, self.tensor(embeddings)), 0)

        # Update id offset
        self.config["offset"] += new

    def delete(self, ids):
        # Filter any index greater than size of array
        ids = [x for x in ids if x < self.backend.shape[0]]

        # Clear specified ids
        self.backend[ids] = self.tensor(torch.zeros((len(ids), self.backend.shape[1])))

    def search(self, queries, limit):
        # Dot product on normalized vectors is equal to cosine similarity
        scores = torch.mm(self.tensor(queries), self.backend.T).tolist()

        # Add index and sort desc based on score
        return [sorted(enumerate(score), key=lambda x: x[1], reverse=True)[:limit] for score in scores]

    def count(self):
        # Get count of non-zero rows (ignores deleted rows)
        return self.backend[~torch.all(self.backend == 0, axis=1)].shape[0]

    def save(self, path):
        # Save array to file
        with open(path, "wb") as handle:
            pickle.dump(self.backend, handle, protocol=__pickle__)

    def tensor(self, array):
        """
        Loads array as a Tensor. Loads to GPU device, if available.

        Args:
            array: data array

        Returns:
            Tensor
        """

        # Convert array to Tensor
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)

        # Load to GPU device, if available
        return array.cuda() if torch.cuda.is_available() else array
