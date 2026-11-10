"""
Annoy ANN module tests
"""

import numpy as np

from txtai import Embeddings

from .base import DenseTest


class TestAnnoy(DenseTest):
    """
    Annoy ANN tests.
    """

    def testAnnoy(self):
        """
        Test Annoy backend
        """

        self.runTests("annoy", None, False)

    def testAnnoyCustom(self):
        """
        Test Annoy backend with custom settings
        """

        # Test with custom settings
        self.runTests("annoy", {"annoy": {"ntrees": 2, "searchk": 1}}, False)

    def testAnnoyDelete(self):
        """
        Test that a delete with an Annoy backend leaves the database unchanged
        """

        embeddings = Embeddings(method="external", transform=lambda x: np.random.rand(len(x), 8).astype(np.float32), content=True, backend="annoy")
        embeddings.index([(x, f"text {x}") for x in range(5)])

        with self.assertRaises(NotImplementedError):
            embeddings.delete([1])

        self.assertEqual(len(embeddings.search("select id from txtai limit 10")), 5)
