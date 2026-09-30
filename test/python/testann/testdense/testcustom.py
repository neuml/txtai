"""
Custom ANN module tests
"""

from txtai.ann import ANNFactory, ANN

from .base import DenseTest


class TestCustom(DenseTest):
    """
    Custom ANN tests.
    """

    def testCustomBackend(self):
        """
        Test resolving a custom backend
        """

        self.runTests("txtai.ann.Faiss")

    def testCustomBackendInvalid(self):
        """
        Test resolving an invalid backend
        """

        with self.assertRaises(ImportError):
            ANNFactory.create({"backend": "pprint.pprint"})

    def testCustomBackendNotFound(self):
        """
        Test resolving an unresolvable backend
        """

        with self.assertRaises(ImportError):
            ANNFactory.create({"backend": "notfound.ann"})

    def testNotImplemented(self):
        """
        Test exceptions for non-implemented methods
        """

        ann = ANN({})

        self.assertRaises(NotImplementedError, ann.load, None)
        self.assertRaises(NotImplementedError, ann.index, None)
        self.assertRaises(NotImplementedError, ann.append, None)
        self.assertRaises(NotImplementedError, ann.delete, None)
        self.assertRaises(NotImplementedError, ann.search, None, None)
        self.assertRaises(NotImplementedError, ann.count)
        self.assertRaises(NotImplementedError, ann.save, None)
