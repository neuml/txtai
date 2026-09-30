"""
IVFSparse ANN module tests
"""

import os
import tempfile

from unittest.mock import patch

import numpy as np

from sklearn.cluster import MiniBatchKMeans

from txtai.ann import SparseANNFactory

from .base import SparseTest


class TestIVFSparse(SparseTest):
    """
    IVFSparse ANN tests.
    """

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

        # Deleting the same id again or an id that was never indexed is a no-op for the count
        ann.delete([0])
        ann.delete([count + 100])
        self.assertEqual(ann.count(), count - 1)

        # Re-validate search results
        results = [x[0] for x in ann.search(append[0], 10)[0]]
        self.assertIn(insert.shape[0], results)

        # Save and reload index with deletes
        ann.save(path)
        ann.load(path)
        self.assertEqual(ann.count(), count - 1)

        # Close ANN
        ann.close()

        # Test cluster pruning
        ann = SparseANNFactory.create({"backend": "ivfsparse", "ivfsparse": {"nlist": 15, "nprobe": 1, "sample": 1.0}})
        ann.index(insert)
        self.assertLessEqual(len(ann.blocks), 15)
        ann.close()

    def testIVFSparseDeleteArray(self):
        """
        Test IVFSparse with ids deleted using a NumPy array
        """

        # Generate test record
        data = self.generate(50, 30522)

        # Create ANN
        path = os.path.join(tempfile.gettempdir(), "ivfsparse.deletes")
        ann = SparseANNFactory.create({"backend": "ivfsparse"})
        ann.index(data)

        # Test delete with ids passed in as a NumPy array
        ann.delete(np.array([0, 1]))
        self.assertEqual(ann.count(), 48)

        # Validate save/load
        ann.save(path)
        ann.load(path)
        self.assertEqual(ann.count(), 48)

        # Close ANN
        ann.close()

    def testIVFSparseSortOrder(self):
        """
        Test IVFSparse returns results sorted by score descending
        """

        # Generate test data
        data = self.generate(50, 30522)

        ann = SparseANNFactory.create({"backend": "ivfsparse"})
        ann.index(data)

        # Each result list must be ranked by score descending
        for results in ann.search(data[:5], 10):
            scores = [score for _, score in results]
            self.assertEqual(scores, sorted(scores, reverse=True))

        ann.close()

    def testIVFSparseTopnOverLimit(self):
        """
        Test IVFSparse topn when limit exceeds the number of indexed documents
        """

        # Generate a small dataset (5 documents)
        data = self.generate(5, 30522)

        ann = SparseANNFactory.create({"backend": "ivfsparse"})
        ann.index(data)

        # Search with limit (10) greater than document count (5)
        results = ann.search(data[0], 10)
        self.assertGreater(len(results[0]), 0)

        # Batch search with multiple queries exceeding document count
        results = ann.search(data, 10)
        self.assertEqual(len(results), data.shape[0])
        for result in results:
            self.assertGreater(len(result), 0)

        ann.close()

    def testIVFSparseNFeatures(self):
        """
        Test IVFSparse nfeatures setting limits model training to the top n features
        """

        # Generate test data
        data = self.generate(500, 100)

        # Capture the feature count passed to model training
        features = []
        fit = MiniBatchKMeans.fit

        def logfit(self, x, *args, **kwargs):
            features.append(x.shape[1])
            return fit(self, x, *args, **kwargs)

        # nfeatures unset trains on the full feature space
        ann = SparseANNFactory.create({"backend": "ivfsparse", "ivfsparse": {"nlist": 2}})
        with patch("txtai.ann.sparse.ivfsparse.MiniBatchKMeans.fit", logfit):
            ann.index(data)
        ann.close()

        # nfeatures limits training to the top n features
        ann = SparseANNFactory.create({"backend": "ivfsparse", "ivfsparse": {"nlist": 2, "nfeatures": 10}})
        with patch("txtai.ann.sparse.ivfsparse.MiniBatchKMeans.fit", logfit):
            ann.index(data)
        ann.close()

        self.assertEqual(features, [100, 10])
