"""
Sparse ANN module tests
"""

import os
import tempfile
import unittest

from unittest.mock import patch

from scipy.sparse import random
from sklearn.preprocessing import normalize

from txtai.ann import SparseANNFactory


class TestSparse(unittest.TestCase):
    """
    Sparse ANN tests.
    """

    def testCustomBackend(self):
        """
        Test resolving a custom backend
        """

        self.assertIsNotNone(SparseANNFactory.create({"backend": "txtai.ann.IVFSparse"}))

    def testCustomBackendNotFound(self):
        """
        Test resolving an unresolvable backend
        """

        with self.assertRaises(ImportError):
            SparseANNFactory.create({"backend": "notfound.ann"})

    def testIVFSparse(self):
        """
        Test IVFSparse backend
        """

        # Generate test record
        insert = self.generate(500, 30522)
        append = self.generate(500, 30522)

        # Count of records
        count = insert.shape[0] + append.shape[0]

        # Create ANN
        path = os.path.join(tempfile.gettempdir(), "ivfsparse")
        ann = SparseANNFactory.create({"backend": "ivfsparse", "ivfsparse": {"nlist": 2, "nprobe": 2, "sample": 1.0}})

        # Test indexing
        ann.index(insert)
        ann.append(append)

        # Validate search results
        results = [x[0] for x in ann.search(insert[5], 10)[0]]
        self.assertIn(5, results)

        # Validate save/load/delete
        ann.save(path)
        ann.load(path)

        # Validate count
        self.assertEqual(ann.count(), count)

        # Test delete
        ann.delete([0])
        self.assertEqual(ann.count(), count - 1)

        # Re-validate search results
        results = [x[0] for x in ann.search(append[0], 10)[0]]
        self.assertIn(insert.shape[0], results)

        # Close ANN
        ann.close()

    @patch("sqlalchemy.orm.Query.limit")
    def testPGSparse(self, query):
        """
        Test Sparse Postgres backend
        """

        # Generate test record
        data = self.generate(1, 240)

        # Mock database query
        query.return_value = [(x, -1.0) for x in range(data.shape[0])]

        # Create ANN
        path = os.path.join(tempfile.gettempdir(), "pgsparse.sqlite")
        ann = SparseANNFactory.create({"backend": "pgsparse", "dimensions": 240, "pgsparse": {"url": f"sqlite:///{path}", "schema": "txtai"}})

        # Test indexing
        ann.index(data)
        ann.append(data)

        # Validate search results
        self.assertEqual(ann.search(data, 1), [[(0, 1.0)]])

        # Validate save/load/delete
        ann.save(None)
        ann.load(None)

        # Validate count
        self.assertEqual(ann.count(), 2)

        # Test delete
        ann.delete([0])
        self.assertEqual(ann.count(), 1)

        # Close ANN
        ann.close()

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
