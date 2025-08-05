"""
SparseVectors module
"""

# Conditional import
try:
    from scipy.sparse import csr_matrix, vstack
    from sklearn.preprocessing import normalize
    from sklearn.utils.extmath import safe_sparse_dot

    SPARSE = True
except ImportError:
    SPARSE = False

from ...util import SparseArray
from ..base import Vectors


# pylint: disable=W0223
class SparseVectors(Vectors):
    """
    Base class for sparse vector models. Vector models transform input content into sparse arrays.
    """

    def __init__(self, config, scoring, models):
        # Check before parent constructor since it calls loadmodel
        if not SPARSE:
            raise ImportError('SparseVectors is not available - install "vectors" extra to enable')

        super().__init__(config, scoring, models)

        # Get normalization setting
        self.isnormalize = self.config.get("normalize", self.defaultnormalize()) if self.config else None

    def encode(self, data, category=None):
        # Encode data to embeddings
        embeddings = super().encode(data, category)

        # Get sparse torch vector attributes
        embeddings = embeddings.cpu().coalesce()
        indices = embeddings.indices().numpy()
        values = embeddings.values().numpy()

        # Return as SciPy CSR Matrix
        return csr_matrix((values, indices), shape=embeddings.size())

    def vectors(self, documents, batchsize=500, checkpoint=None, buffer=None, dtype=None):
        # Run indexing
        ids, dimensions, batches, stream = self.index(documents, batchsize, checkpoint)

        # Rebuild sparse array
        embeddings = None
        with open(stream, "rb") as queue:
            for _ in range(batches):
                # Read in array batch
                data = self.loadembeddings(queue)
                embeddings = vstack((embeddings, data)) if embeddings is not None else data

        # Return sparse array
        return (ids, dimensions, embeddings)

    def dot(self, queries, data):
        return safe_sparse_dot(queries, data.T, dense_output=True).tolist()

    def loadembeddings(self, f):
        return SparseArray().load(f)

    def saveembeddings(self, f, embeddings):
        SparseArray().save(f, embeddings)

    def truncate(self, embeddings):
        raise ValueError("Truncate is not supported for sparse vectors")

    def normalize(self, embeddings):
        # Optionally normalize embeddings using method that supports sparse vectors
        return normalize(embeddings, copy=False) if self.isnormalize else embeddings

    def quantize(self, embeddings):
        raise ValueError("Quantize is not supported for sparse vectors")

    def defaultnormalize(self):
        """
        Returns the default normalization setting.

        Returns:
            default normalization setting
        """

        # Sparse vector embeddings typically perform better as unnormalized
        return False
