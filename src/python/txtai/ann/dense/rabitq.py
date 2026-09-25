"""
RabitQ module
"""

import math
import os
import tempfile

# Conditional import
try:
    from rabitqlib import HnswIndex, IvfIndex
    from sklearn.cluster import KMeans

    RABITQ = True
except ImportError:
    RABITQ = False

from ...archive import ArchiveFactory
from ..base import ANN

# Core library imports
from ...util import Library

library = Library()
np = library.numpy()
safetensors = library.safetensors()

# Sentinel for unfilled top-k slots in native search results
SENTINEL = 0xFFFFFFFF

# External id for rows marked as deleted
DELETED = -1


class RabitQ(ANN):
    """
    Builds an ANN index using the rabitqlib library (RaBitQ quantization).
    """

    def __init__(self, config):
        super().__init__(config)

        if not RABITQ:
            raise ImportError('rabitqlib is not available - install "ann" extra to enable')

        # Retained corpus: native indexes have no append/delete, appends rebuild from these rows and deletes mark rows until the next rebuild
        self.vectors = None
        self.ids = None
        self.numclusters = 0

    def load(self, path):
        self.close()

        try:
            # rabitqlib reads the whole index into memory, so the extracted files are only needed while loading
            with tempfile.TemporaryDirectory() as directory:
                archive = ArchiveFactory.create(directory)
                archive.load(path, "tar")

                # Retained corpus tensors plus index metadata
                with safetensors.safe_open(os.path.join(directory, "vectors.safetensors"), framework="np") as f:
                    metadata = f.metadata() if f.metadata() else {}
                    self.vectors = np.ascontiguousarray(f.get_tensor("vectors"), dtype=np.float32)
                    self.ids = np.array(f.get_tensor("ids"), dtype=np.int64)

                self.numclusters = int(metadata.get("clusters", 0))

                # Mode comes from the current config, falling back to the saved snapshot. Invalid modes raise ValueError before native code runs.
                mode = self.mode(metadata.get("mode", "ivf"))

                # Empty indexes are saved without a native index file
                native = os.path.join(directory, "index.bin")
                if os.path.exists(native):
                    self.backend = IvfIndex.load(native) if mode == "ivf" else HnswIndex.load(native)

            # Default the offset for snapshots that don't have it
            self.config["offset"] = int(metadata.get("offset", len(self.ids)))
        except Exception:
            self.close()
            raise

    def index(self, embeddings):
        self.close()

        # Mode is validated before any state is stored
        mode = self.mode()

        self.vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.ids = np.arange(self.vectors.shape[0], dtype=np.int64)

        self.build()

        # Add id offset and index build metadata
        self.config["offset"] = self.vectors.shape[0]
        self.metadata(self.settings(mode))

    def append(self, embeddings):
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        new = embeddings.shape[0]

        self.vectors = np.ascontiguousarray(np.concatenate((self.vectors, embeddings), axis=0), dtype=np.float32)
        self.ids = np.concatenate((self.ids, np.arange(self.config["offset"], self.config["offset"] + new, dtype=np.int64)))

        self.build()

        # Update id offset and index metadata
        self.config["offset"] += new
        self.metadata()

    def delete(self, ids):
        if not ids or not self.count():
            return

        # Mark rows as deleted, search skips them. Unknown ids match nothing and are silently ignored.
        self.ids[np.isin(self.ids, np.array(ids, dtype=np.int64))] = DELETED

        # Rebuild once deleted rows outnumber live rows, this bounds search overfetch
        if self.count() < len(self.ids) - self.count():
            self.build()

    def search(self, queries, limit):
        count = self.count()
        if not count:
            return [[] for _ in queries]

        queries = np.ascontiguousarray(queries, dtype=np.float32)

        # Overfetch by the number of deleted rows, clamped to the native row count as the native index errors when k exceeds it
        k = min(limit + len(self.ids) - count, len(self.ids))

        mode = self.mode()
        if mode == "ivf":
            nprobe = self.setting("nprobe", max(1, round(self.numclusters / 16)))
            ids, distances = self.backend.search(queries, k, nprobe)
        else:
            ef = self.setting("efsearch", None)
            ids, distances = self.backend.search(queries, k, ef or 0)

        # Map results to [(id, score)], dropping unfilled sentinel slots and deleted rows
        results = []
        for pids, dists in zip(ids.tolist(), distances.tolist()):
            result = [(int(self.ids[pid]), float(1.0 - dist)) for pid, dist in zip(pids, dists) if pid != SENTINEL and self.ids[pid] != DELETED]
            results.append(result[:limit])

        return results

    def count(self):
        return int(np.count_nonzero(self.ids != DELETED)) if self.ids is not None else 0

    def save(self, path):
        # Stage the native index and the retained corpus, then bundle both into a single tar file
        with tempfile.TemporaryDirectory() as directory:
            # Empty indexes have no native index to save
            if self.backend is not None:
                self.backend.save(os.path.join(directory, "index.bin"))

            # Save retained corpus tensors, safetensors metadata values must be strings
            safetensors.numpy.save_file(
                {"vectors": self.vectors, "ids": self.ids},
                os.path.join(directory, "vectors.safetensors"),
                {"mode": self.mode(), "clusters": str(self.numclusters), "offset": str(self.config.get("offset", len(self.ids)))},
            )

            archive = ArchiveFactory.create(directory)
            archive.save(path, "tar")

    def close(self):
        # Parent logic releases the backend
        super().close()

        self.vectors = None
        self.ids = None
        self.numclusters = 0

    def mode(self, default="ivf"):
        """
        Returns the configured index mode.

        Args:
            default: default value when the mode setting is not found

        Returns:
            index mode, only "ivf" or "hnsw" are accepted
        """

        mode = self.setting("mode", default)
        if mode not in ("ivf", "hnsw"):
            raise ValueError(f'Invalid RabitQ mode: {mode}. Supported modes are "ivf" and "hnsw".')

        return mode

    def settings(self, mode):
        """
        Returns index build settings for metadata.

        Args:
            mode: index mode

        Returns:
            dict of build settings
        """

        if mode == "ivf":
            return {
                "mode": mode,
                "nbits": self.setting("nbits", 1),
                "clusters": self.numclusters,
                "nprobe": self.setting("nprobe", max(1, round(self.numclusters / 16))),
            }

        return {
            "mode": mode,
            "nbits": self.setting("nbits", 1),
            "m": self.setting("m", 16),
            "efconstruction": self.setting("efconstruction", 200),
            "efsearch": self.setting("efsearch", None),
            "randomseed": self.setting("randomseed", 100),
        }

    def build(self):
        """
        Builds the native index from the retained corpus, dropping rows marked as deleted.
        """

        live = self.ids != DELETED
        if not live.all():
            self.vectors = np.ascontiguousarray(self.vectors[live], dtype=np.float32)
            self.ids = self.ids[live]

        rows = self.vectors.shape[0]
        if not rows:
            self.backend = None
            self.numclusters = 0
            return

        mode = self.mode()
        dim = self.config["dimensions"]

        # Quantization bits per dimension, validated by rabitqlib
        nbits = self.setting("nbits", 1)

        # Default clusters to 4 * sqrt(N), bounded to [1, N] so KMeans always fits
        clusters = self.setting("clusters", None)
        self.numclusters = max(1, min(clusters if clusters else round(4 * math.sqrt(rows)), rows))

        cluster = KMeans(n_clusters=self.numclusters, random_state=0, n_init=1).fit(self.vectors)
        centroids = np.ascontiguousarray(cluster.cluster_centers_, dtype=np.float32)
        clusterids = np.ascontiguousarray(cluster.labels_, dtype=np.uint32)

        if mode == "ivf":
            self.backend = IvfIndex(dim=dim, max_elements=rows, num_clusters=self.numclusters, nbits=nbits, metric="ip")
        else:
            self.backend = HnswIndex(
                dim=dim,
                max_elements=rows,
                M=self.setting("m", 16),
                ef_construction=self.setting("efconstruction", 200),
                nbits=nbits,
                metric="ip",
                random_seed=self.setting("randomseed", 100),
            )

        self.backend.build(self.vectors, centroids, clusterids)
