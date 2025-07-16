"""
PGSparse module
"""

import os

import numpy as np

# Conditional import
try:
    from pgvector import SparseVector
    from pgvector.sqlalchemy import SPARSEVEC

    PGSPARSE = True
except ImportError:
    PGSPARSE = False

from ..dense import PGVector


class PGSparse(PGVector):
    """
    Builds a Sparse ANN index backed by a Postgres database.
    """

    def __init__(self, config):
        if not PGSPARSE:
            raise ImportError('PGSparse is not available - install "ann" extra to enable')

        super().__init__(config)

        # Quantization not supported
        self.qbits = None

    def defaulttable(self):
        return "svectors"

    def url(self):
        return self.setting("url", os.environ.get("SCORING_URL", os.environ.get("ANN_URL")))

    def column(self):
        return SPARSEVEC(self.config["dimensions"])

    def operation(self):
        return "sparsevec_ip_ops"

    def prepare(self, data):
        # pgvector only allows 1000 non-zero values for sparse vectors
        # Trim to top 1000 values, if necessary
        if data.count_nonzero() > 1000:
            value = -np.sort(-data[0, :].data)[1000]
            data.data = np.where(data.data > value, data.data, 0)
            data.eliminate_zeros()

        # Wrap as sparse vector
        return SparseVector(data)
