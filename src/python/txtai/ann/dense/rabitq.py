"""
RabitQ module
"""

import math

# Conditional import
try:
    from rabitqlib import HnswIndex, IvfIndex
    from sklearn.cluster import KMeans

    RABITQ = True
except ImportError:
    RABITQ = False

from ..base import ANN

# Core library imports
from ...util import Library

library = Library()
np = library.numpy()

# Sentinel for unfilled top-k slots in native search results
SENTINEL = 0xFFFFFFFF


class RabitQ(ANN):
    """
    Builds an ANN index using the rabitqlib library (RaBitQ quantization).

    Only the native index is stored, as a single file. No vectors are kept outside of it. ivf indexes support append
    and delete natively. The native id of a row is its index id. hnsw indexes can't be extended or have rows removed,
    so append and delete are not supported in that mode.
    """

    def __init__(self, config):
        super().__init__(config)

        if not RABITQ:
            raise ImportError('rabitqlib is not available - install "ann" extra to enable')

        # Number of rows removed from the native index (ivf mode only)
        self.config["deletes"] = self.config.get("deletes", 0)

    def load(self, path):
        self.close()

        # Invalid modes raise ValueError before native code runs
        self.backend = IvfIndex.load(path) if self.mode() == "ivf" else HnswIndex.load(path)

    def index(self, embeddings):
        self.close()

        # Mode is validated before any state is stored
        mode = self.mode()

        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        rows = vectors.shape[0]

        # Quantization bits per dimension, validated by rabitqlib
        nbits = self.setting("nbits", 1)

        # Default clusters to 4 * sqrt(N), bounded to [1, N] so KMeans always fits
        clusters = self.setting("clusters", None)
        clusters = max(1, min(clusters if clusters else round(4 * math.sqrt(rows)), rows))

        cluster = KMeans(n_clusters=clusters, random_state=0, n_init=1).fit(vectors)
        centroids = np.ascontiguousarray(cluster.cluster_centers_, dtype=np.float32)
        clusterids = np.ascontiguousarray(cluster.labels_, dtype=np.uint32)

        dim = self.config["dimensions"]
        if mode == "ivf":
            backend = IvfIndex(dim=dim, max_elements=rows, num_clusters=clusters, nbits=nbits, metric="ip")
            settings = {"mode": mode, "nbits": nbits, "clusters": clusters, "nprobe": self.setting("nprobe", self.nprobe(clusters))}
        else:
            backend = HnswIndex(
                dim=dim,
                max_elements=rows,
                M=self.setting("m", 16),
                ef_construction=self.setting("efconstruction", 200),
                nbits=nbits,
                metric="ip",
                random_seed=self.setting("randomseed", 100),
            )
            settings = {
                "mode": mode,
                "nbits": nbits,
                "m": self.setting("m", 16),
                "efconstruction": self.setting("efconstruction", 200),
                "efsearch": self.setting("efsearch", None),
                "randomseed": self.setting("randomseed", 100),
            }

        backend.build(vectors, centroids, clusterids)
        self.backend = backend

        # Add id offset and index build metadata
        self.config["offset"] = rows
        self.config["deletes"] = 0
        self.metadata(settings)

    def append(self, embeddings):
        if self.mode() != "ivf":
            super().append(embeddings)

        # Native ids continue from the last row ever added
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.backend.add(embeddings)

        # Update id offset and index metadata
        self.config["offset"] = self.backend.max_elements
        self.metadata()

    def delete(self, ids):
        if self.mode() != "ivf":
            super().delete(ids)

        # Native remove hides rows from search. Unknown ids are silently ignored.
        ids = [uid for uid in ids if 0 <= uid < self.backend.max_elements]
        if ids:
            self.config["deletes"] += self.backend.remove(np.array(ids, dtype=np.int64))

    def search(self, queries, limit):
        if not self.count():
            return [[] for _ in queries]

        queries = np.ascontiguousarray(queries, dtype=np.float32)

        # The native index errors when k exceeds its row count
        k = min(limit, self.count())

        if self.mode() == "ivf":
            nprobe = self.setting("nprobe", self.nprobe(self.backend.num_clusters))
            ids, distances = self.backend.search(queries, k, nprobe)
        else:
            ids, distances = self.backend.search(queries, k, self.setting("efsearch", None) or 0)

        # Map results to [(id, score)], dropping unfilled sentinel slots
        return [
            [(int(uid), float(1.0 - dist)) for uid, dist in zip(uids, dists) if uid != SENTINEL]
            for uids, dists in zip(ids.tolist(), distances.tolist())
        ]

    def count(self):
        return self.backend.max_elements - self.config["deletes"] if self.backend is not None else 0

    def save(self, path):
        self.backend.save(path)

    def mode(self):
        """
        Returns the configured index mode.

        Returns:
            index mode, only "ivf" or "hnsw" are accepted
        """

        mode = self.setting("mode", "ivf")
        if mode not in ("ivf", "hnsw"):
            raise ValueError(f'Invalid RabitQ mode: {mode}. Supported modes are "ivf" and "hnsw".')

        return mode

    def nprobe(self, clusters):
        """
        Returns the default number of clusters to probe at search time.

        Args:
            clusters: number of IVF clusters

        Returns:
            default nprobe
        """

        return max(1, round(clusters / 16))
