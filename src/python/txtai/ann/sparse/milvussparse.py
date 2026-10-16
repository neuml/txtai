"""
MilvusSparse module
"""

import math

# Conditional import
try:
    import milvus_lite
except ImportError:
    pass

from ..dense import Milvus


class MilvusSparse(Milvus):
    """
    Builds a Sparse ANN index using the milvus-lite library. Requires milvus-lite >= 3.2.2 for inner product sparse search.

    milvus-lite only stores finite, non-negative sparse weights. Invalid vectors raise a ValueError before any data is written.
    """

    def search(self, queries, limit):
        results = []
        for query in queries:
            # Empty queries have no matches
            query = self.prepare(query)
            matches = self.collection.search([query], top_k=limit, metric_type="IP", anns_field="embedding", output_fields=["id"])[0] if query else []
            results.append([(int(match["id"]), float(match["distance"])) for match in matches])

        return results

    def field(self):
        # Empty rows are stored as null vectors to keep ids aligned
        return milvus_lite.FieldSchema("embedding", milvus_lite.DataType.SPARSE_FLOAT_VECTOR, nullable=True)

    def indexparams(self, m):
        return {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "params": {}}

    def prepare(self, data):
        vector = {}
        for index, value in zip(data.indices, data.data):
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"milvussparse only supports finite, non-negative weights, found {value} at index {index}")

            # Skip explicit zeros
            if value:
                vector[int(index)] = value

        # milvus-lite requires at least one non-zero weight, store empty rows as null
        return vector if vector else None
