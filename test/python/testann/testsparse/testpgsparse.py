"""
PGSparse ANN module tests
"""

import os
import tempfile

from unittest.mock import patch

from scipy.sparse import random

from txtai.ann import SparseANNFactory

from .base import SparseTest


class TestPGSparse(SparseTest):
    """
    PGSparse ANN tests.
    """

    @patch("sqlalchemy.orm.Query.limit")
    def testPGSparse(self, query):
        """
        Test Sparse Postgres backend
        """

        # Generate test record
        data = self.generate(1, 30522)

        # Mock database query
        query.return_value = [(x, -1.0) for x in range(data.shape[0])]

        # Create ANN
        path = os.path.join(tempfile.gettempdir(), "pgsparse.sqlite")
        ann = SparseANNFactory.create({"backend": "pgsparse", "dimensions": 30522, "pgsparse": {"url": f"sqlite:///{path}", "schema": "txtai"}})

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

        # Test > 1000 dimensions
        data = random(1, 30522, format="csr", density=0.1)
        ann.index(data)
        self.assertEqual(ann.count(), 1)

        # Close ANN
        ann.close()
