"""
Annoy ANN module tests
"""

from txtai.embeddings import Embeddings

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

    def testAnnoyImplements(self):
        """
        Test that Annoy reports appends and deletes as not supported
        """

        model = self.backend("annoy")
        model.implements("search")

        for method in ["append", "delete"]:
            with self.assertRaisesRegex(NotImplementedError, f"{method} is not supported by the annoy backend"):
                model.implements(method)

    def testAnnoySubindex(self):
        """
        Test that upserts and deletes fail before changing any data when a subindex backend doesn't support them
        """

        embeddings = Embeddings(
            {
                "defaults": False,
                "content": True,
                "indexes": {"index1": {"path": "sentence-transformers/nli-mpnet-base-v2", "backend": "annoy"}},
            }
        )
        embeddings.index([(0, "hello world"), (1, "foo bar")])

        # Check that nothing is changed when the subindex backend can't append or delete
        rows = [{"id": "0", "text": "hello world"}, {"id": "1", "text": "foo bar"}]
        query = "select id, text from txtai order by id"

        with self.assertRaisesRegex(NotImplementedError, "is not supported by the annoy backend"):
            embeddings.upsert([(0, "brand new text"), (2, "another doc")])

        self.assertEqual(embeddings.count(), 2)
        self.assertEqual(embeddings.search(query), rows)

        with self.assertRaisesRegex(NotImplementedError, "delete is not supported by the annoy backend"):
            embeddings.delete([0])

        self.assertEqual(embeddings.count(), 2)
        self.assertEqual(embeddings.search(query), rows)
