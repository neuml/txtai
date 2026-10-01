"""
ZvecSparse module
"""

# Conditional import
try:
    import zvec
except ImportError:
    pass

from ..dense import Zvec


class ZvecSparse(Zvec):
    """
    Builds a Sparse ANN index using the zvec library.
    """

    def search(self, queries, limit):
        # Lookup search settings
        param = zvec.HnswQueryParam(ef=self.setting("efsearch", 300))

        results = []
        for query in queries:
            # zvec requires at least one non-zero value per query
            matches = self.backend.query(zvec.Query(field_name="embedding", vector=self.prepare(query), param=param), topk=limit) if query.nnz else []
            results.append([(int(match.id), float(match.score)) for match in matches])

        return results

    def datatype(self):
        return zvec.DataType.SPARSE_VECTOR_FP32

    def prepare(self, data):
        return {int(index): float(value) for index, value in zip(data.indices, data.data)}
