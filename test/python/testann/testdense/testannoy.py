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

    def testAnnoyModify(self):
        """
        Test that deletes and upserts with an Annoy backend raise an error and leave the database unchanged
        """

        embeddings = Embeddings(method="external", transform=lambda x: np.random.rand(len(x), 8).astype(np.float32), content=True, backend="annoy")
        embeddings.index([(x, f"text {x}") for x in range(5)])

        with self.assertRaisesRegex(NotImplementedError, "delete is not supported by the annoy backend"):
            embeddings.delete([1])

        with self.assertRaisesRegex(NotImplementedError, "append is not supported by the annoy backend"):
            embeddings.upsert([(1, "updated text"), (5, "new text")])

        self.assertEqual(embeddings.search("select id, text from txtai where id = 1")[0]["text"], "text 1")
        self.assertEqual(len(embeddings.search("select id from txtai limit 10")), 5)
