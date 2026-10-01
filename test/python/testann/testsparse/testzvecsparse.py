"""
ZvecSparse ANN module tests
"""

import os
import tempfile

from scipy.sparse import csr_matrix

from txtai.ann import SparseANNFactory

from .base import SparseTest


class TestZvecSparse(SparseTest):
    """
    ZvecSparse ANN tests.
    """

    def testZvecSparse(self):
        """
        Test Sparse zvec backend
        """

        # Generate test record
        data = self.generate(1, 30522)

        # Create ANN
        path = os.path.join(tempfile.gettempdir(), "zvecsparse")
        ann = SparseANNFactory.create({"backend": "zvecsparse", "dimensions": 30522})

        # Test indexing
        ann.index(data)
        ann.append(data)

        # Validate search results
        self.assertEqual([[(uid, round(score, 5)) for uid, score in x] for x in ann.search(data, 1)], [[(0, 1.0)]])

        # Validate save/load
        ann.save(path)
        ann.load(path)

        # Validate count
        self.assertEqual(ann.count(), 2)

        # Test delete
        ann.delete([0])
        self.assertEqual(ann.count(), 1)
        self.assertEqual(ann.search(data, 1)[0][0][0], 1)

        # Close ANN
        ann.close()

    def testZvecSparseEmpty(self):
        """
        Test ZvecSparse with an empty query
        """

        # Create ANN with an empty row
        data = csr_matrix([[1.0, 0.0], [0.0, 0.0]])
        ann = SparseANNFactory.create({"backend": "zvecsparse", "dimensions": 2})
        ann.index(data)

        # Validate empty rows are indexed and searching with an empty query doesn't raise an error
        self.assertEqual(ann.count(), 2)
        self.assertEqual(len(ann.search(data, 1)), 2)

        # Close ANN
        ann.close()

    def testZvecSparseScores(self):
        """
        Test ZvecSparse scores are exact inner products
        """

        # Create ANN
        data = self.generate(50, 30522)
        ann = SparseANNFactory.create({"backend": "zvecsparse", "dimensions": 30522})
        ann.index(data)

        # Validate scores against exact inner products
        expected = (data[:5] @ data.T).toarray()
        for x, results in enumerate(ann.search(data[:5], 10)):
            for uid, score in results:
                self.assertAlmostEqual(score, expected[x, uid], places=5)

        # Close ANN
        ann.close()
