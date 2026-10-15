"""
MilvusSparse ANN module tests
"""

import os
import tempfile

from unittest.mock import patch

from scipy.sparse import csr_matrix

from txtai.ann import SparseANNFactory

from .base import SparseTest


class TestMilvusSparse(SparseTest):
    """
    MilvusSparse ANN tests.
    """

    def testMilvusSparse(self):
        """
        Test Sparse milvus-lite backend
        """

        # Generate test record
        data = self.generate(1, 30522)

        # Create ANN
        path = os.path.join(tempfile.gettempdir(), "milvussparse")
        ann = SparseANNFactory.create({"backend": "milvussparse", "dimensions": 30522})

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

    def testMilvusSparseEmpty(self):
        """
        Test MilvusSparse with an empty query
        """

        # Create ANN with an empty row
        data = csr_matrix([[1.0, 0.0], [0.0, 0.0]])
        ann = SparseANNFactory.create({"backend": "milvussparse", "dimensions": 2})
        ann.index(data)

        # Validate empty rows are indexed and searching with an empty query doesn't raise an error
        self.assertEqual(ann.count(), 2)
        self.assertEqual(len(ann.search(data, 1)), 2)

        # Close ANN
        ann.close()

    def testMilvusSparseScores(self):
        """
        Test MilvusSparse scores are exact inner products
        """

        # Create ANN
        data = self.generate(50, 30522)
        ann = SparseANNFactory.create({"backend": "milvussparse", "dimensions": 30522})
        ann.index(data)

        # Validate scores against exact inner products
        expected = (data[:5] @ data.T).toarray()
        for x, results in enumerate(ann.search(data[:5], 10)):
            for uid, score in results:
                self.assertAlmostEqual(score, expected[x, uid], places=5)

        # Close ANN
        ann.close()

    def testMilvusSparseInvalid(self):
        """
        Test MilvusSparse rejects invalid vectors without modifying the index
        """

        # Create ANN
        ann = SparseANNFactory.create({"backend": "milvussparse", "dimensions": 2})
        ann.index(csr_matrix([[1.0, 0.0]]))

        # Validate negative and non-finite weights are rejected before any data is written
        for value in [-1.0, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                ann.append(csr_matrix([[0.0, 1.0], [0.0, value]]))

            self.assertEqual(ann.count(), 1)
            self.assertEqual(ann.config["offset"], 1)

        # Validate a valid append after a rejected append uses the next id
        ann.append(csr_matrix([[0.0, 1.0]]))
        self.assertEqual(ann.search(csr_matrix([[0.0, 1.0]]), 1), [[(1, 1.0)]])

        # Close ANN
        ann.close()

    def testMilvusSparseRollback(self):
        """
        Test MilvusSparse rolls back a partially inserted batch
        """

        # Create ANN
        ann = SparseANNFactory.create({"backend": "milvussparse", "dimensions": 2})
        ann.index(csr_matrix([[1.0, 0.0]]))

        # Fail the second insert batch after the first batch is written
        insert, batches = ann.collection.insert, []

        def failsecond(records):
            batches.append(records)
            if len(batches) > 1:
                raise RuntimeError("insert failed")

            return insert(records)

        with patch.object(ann.collection, "insert", side_effect=failsecond):
            with self.assertRaises(RuntimeError):
                ann.append(csr_matrix([[0.0, 1.0]] * 1025))

        # Validate the index and offset are unchanged
        self.assertEqual(ann.count(), 1)
        self.assertEqual(ann.config["offset"], 1)

        # Close ANN
        ann.close()
