"""
Embeddings module tests
"""

import tempfile
import unittest

import numpy as np

from txtai.ann import ANN

class TestEmbeddings(unittest.TestCase):
    """
    Embeddings tests
    """

    def setUp(self):
        """
        Initialize test data.
        """

        # Generate random embeddings
        # np.random.rand(x, y)
