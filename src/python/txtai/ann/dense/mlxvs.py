"""
mlxvs module
"""

# Conditional import
try:
    import mlx.core as mx
    import mlxvs

    MLXVS = True
except ImportError:
    MLXVS = False

from ..base import ANN

# Core library imports
from ...util import Library

library = Library()
np = library.numpy()
safetensors = library.safetensors()


class Mlxvs(ANN):
    """
    Builds an ANN index on the Apple silicon GPU using mlxvs, a CAGRA graph index built on MLX.
    """

    def __init__(self, config):
        super().__init__(config)

        if not MLXVS:
            raise ImportError('mlxvs is not available - install "ann" extra to enable (requires macOS on Apple silicon)')

        # Indexed vectors (graph rebuilds and exact search use them) and deleted rows
        self.vectors, self.deleted = None, None

    def load(self, path):
        with safetensors.safe_open(path, framework="np") as f:
            self.vectors, self.deleted = f.get_tensor("vectors"), f.get_tensor("deleted").astype(bool)
            graph = f.get_tensor("graph") if "graph" in f.keys() else None

        self.backend = mlxvs.CagraIndex(self.dataset(), mx.array(graph), "inner_product") if graph is not None else None

    def index(self, embeddings):
        self.vectors = embeddings.astype(np.float32)
        self.deleted = np.zeros(embeddings.shape[0], dtype=bool)
        self.build()

        # Add id offset and index build metadata
        self.config["offset"] = embeddings.shape[0]
        self.metadata(
            {
                "graphdegree": self.setting("graphdegree", 32),
                "intermediategraphdegree": self.setting("intermediategraphdegree", 64),
                "dtype": self.setting("dtype", "float32"),
            }
        )

    def append(self, embeddings):
        # The graph is rebuilt over all rows, which takes seconds per million rows on the GPU
        self.vectors = np.concatenate([self.vectors, embeddings.astype(np.float32)])
        self.deleted = np.concatenate([self.deleted, np.zeros(embeddings.shape[0], dtype=bool)])
        self.build()

        # Update id offset and index metadata
        self.config["offset"] += embeddings.shape[0]
        self.metadata()

    def delete(self, ids):
        # Deleted rows stay in the graph to route the search but are filtered out of results
        ids = np.asarray(ids, dtype=np.int64)
        self.deleted[ids[(ids >= 0) & (ids < self.deleted.shape[0])]] = True

    def search(self, queries, limit):
        count = self.count()
        limit = min(limit, count)
        if not limit:
            return [[] for _ in range(queries.shape[0])]

        queries = queries.astype(np.float32)
        itopk = max(self.setting("itopk", 64), limit)

        # The graph search keeps at most 1024 candidates per query, exact search covers the rest
        if self.backend is None or itopk > 1024:
            ids, scores = self.exact(queries, limit)
        else:
            mask = ~self.deleted if self.deleted.any() else None
            ids, scores = self.backend.search(queries, limit, itopk_size=itopk, filter=mask)
            ids, scores = np.asarray(ids), np.asarray(scores)

        # Map results to [(id, score)], dropping missing results (-1)
        return [[(int(i), float(s)) for i, s in zip(ids[x], scores[x]) if i >= 0] for x in range(queries.shape[0])]

    def count(self):
        return int(self.deleted.shape[0] - self.deleted.sum()) if self.deleted is not None else 0

    def save(self, path):
        data = {"vectors": self.vectors, "deleted": self.deleted.astype(np.uint8)}
        if self.backend is not None:
            data["graph"] = np.asarray(self.backend.graph)

        safetensors.numpy.save_file(data, path)

    def build(self):
        """
        Builds the search graph over the indexed vectors. Small indexes are searched exactly.
        """

        degree, intermediate = self.setting("graphdegree", 32), self.setting("intermediategraphdegree", 64)
        if self.vectors.shape[0] <= intermediate:
            self.backend = None
        else:
            graph = mlxvs.cagra_build(self.vectors, degree, "inner_product", intermediate_graph_degree=intermediate)
            self.backend = mlxvs.CagraIndex(self.dataset(), graph, "inner_product")

    def dataset(self):
        """
        Vectors as stored for search: float32 or float16 (dtype setting).

        Returns:
            mlx array
        """

        dtype = self.setting("dtype", "float32")
        return mx.array(self.vectors).astype(getattr(mx, dtype))

    def exact(self, queries, limit):
        """
        Exact inner product search over the rows not deleted.

        Args:
            queries: query vectors
            limit: maximum results

        Returns:
            ids, scores
        """

        live = np.flatnonzero(~self.deleted)
        scores = queries @ self.vectors[live].T
        order = np.argsort(-scores, axis=1)[:, :limit]
        return live[order], np.take_along_axis(scores, order, axis=1)
