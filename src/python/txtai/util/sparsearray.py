"""
SparseArray module
"""

import numpy as np

# Conditional import
try:
    from scipy.sparse import csr_matrix

    SCIPY = True
except ImportError:
    SCIPY = False


class SparseArray:
    """
    Methods to load and save sparse arrays to file.
    """

    def __init__(self):
        """
        Creates a SparseArray instance.
        """

        if not SCIPY:
            raise ImportError("SciPy is not available - install scipy to enable")

    def load(self, f):
        """
        Loads a sparse array from file.

        Args:
            f: input file handle

        Returns:
            sparse array
        """

        # Load raw data
        data, indices, indptr, shape = (
            np.load(f, allow_pickle=False),
            np.load(f, allow_pickle=False),
            np.load(f, allow_pickle=False),
            np.load(f, allow_pickle=False),
        )

        # Load data into sparse array
        return csr_matrix((data, indices, indptr), shape=shape)

    def save(self, f, array):
        """
        Saves a sparse array to file.

        Args:
            f: output file handle
            array: sparse array
        """

        # Save sparse array to file
        for x in [array.data, array.indices, array.indptr, array.shape]:
            np.save(f, x, allow_pickle=False)
