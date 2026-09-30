"""
Custom ANN module tests
"""

import unittest

from txtai.ann import SparseANNFactory


class TestCustom(unittest.TestCase):
    """
    Custom ANN tests.
    """

    def testCustomBackend(self):
        """
        Test resolving a custom backend
        """

        self.assertIsNotNone(SparseANNFactory.create({"backend": "txtai.ann.IVFSparse"}))

    def testCustomBackendInvalid(self):
        """
        Test resolving an invalid backend
        """

        with self.assertRaises(ImportError):
            SparseANNFactory.create({"backend": "pprint.pprint"})

    def testCustomBackendNotFound(self):
        """
        Test resolving an unresolvable backend
        """

        with self.assertRaises(ImportError):
            SparseANNFactory.create({"backend": "notfound.ann"})
