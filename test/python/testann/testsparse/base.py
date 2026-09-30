"""
Base class for Sparse ANN tests
"""

import unittest

from scipy.sparse import random
from sklearn.preprocessing import normalize


class SparseTest(unittest.TestCase):
    """
    Base class for ANN Sparse tests.
    """

    def generate(self, m, n):
        """
        Generates random normalized sparse data.

        Args:
            m, n: shape of the matrix

        Returns:
            csr matrix
        """

        # Generate random csr matrix
        data = random(m, n, format="csr")

        # Normalize and return
        return normalize(data)
