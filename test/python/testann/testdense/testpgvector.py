"""
PGVector ANN module tests
"""

import os
import tempfile

from unittest.mock import patch

import numpy as np

from sqlalchemy.dialects.postgresql import BIT
from sqlalchemy.ext.compiler import compiles

from txtai.ann import ANNFactory

from .base import DenseTest


class TestPGVector(DenseTest):
    """
    PGVector ANN tests.
    """

    @patch("sqlalchemy.orm.Query.limit")
    def testPGVector(self, query):
        """
        Test PGVector backend
        """

        # pylint: disable=W0613
        @compiles(BIT, "sqlite")
        def compile_bit_sqlite(type_, compiler, **kw):
            return "BLOB"

        # Generate test record
        data = np.random.rand(1, 240).astype(np.float32)

        # Mock database query
        query.return_value = [(x, -1.0) for x in range(data.shape[0])]

        configs = [
            ("full", {"dimensions": 240}, {}, data),
            ("half", {"dimensions": 240}, {"precision": "half"}, data),
            ("binary", {"quantize": 1, "dimensions": 240 * 8}, {}, data.astype(np.uint8)),
        ]

        # Create ANN
        for name, config, pgvector, data in configs:
            path = os.path.join(tempfile.gettempdir(), f"pgvector.{name}.sqlite")
            ann = ANNFactory.create(
                {**{"backend": "pgvector", "pgvector": {**{"url": f"sqlite:///{path}", "schema": "txtai"}, **pgvector}}, **config}
            )

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

            # Test delete with NumPy ids
            ann.delete(np.array([1]))
            self.assertEqual(ann.count(), 0)

            # Close ANN
            ann.close()
