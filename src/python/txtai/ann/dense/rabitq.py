"""
RabitQ module
"""

import math
import os
import shutil
import tempfile

# Conditional import
try:
    from rabitqlib import HnswIndex, IvfIndex

    RABITQ = True
except ImportError:
    RABITQ = False

from ...archive import ArchiveFactory
from ..base import ANN

# Core library imports
from ...util import Library

np = Library().numpy()

# Sentinel for unfilled top-k slots in native search results
SENTINEL = 0xFFFFFFFF


class RabitQ(ANN):
    """
    Builds an ANN index using the rabitqlib library (RaBitQ quantization).
    """

    def __init__(self, config):
        super().__init__(config)

        if not RABITQ:
            raise ImportError('rabitqlib is not available - install "ann" extra to enable')

        self.directory = None

        # Retained corpus: native indexes have no append/delete so both operations rebuild from these rows
        self.vectors = None
        self.ids = None
        self.numclusters = 0

    def load(self, path):
        self.close()
        self.directory = tempfile.mkdtemp()

        try:
            archive = ArchiveFactory.create(self.directory)
            archive.load(path, "tar")

            meta = np.load(os.path.join(self.directory, "meta.npz"), allow_pickle=False)

            self.vectors = np.ascontiguousarray(meta["vectors"], dtype=np.float32)
            self.ids = np.ascontiguousarray(meta["ids"], dtype=np.int64)
            self.numclusters = int(meta["clusters"]) if "clusters" in meta else 0

            # Mode comes from the current config, falling back to the saved snapshot. Invalid modes raise ValueError here, before native code runs.
            mode = self.mode(str(meta["mode"]) if "mode" in meta else "ivf")

            native = os.path.join(self.directory, "index.bin")
            if os.path.exists(native):
                self.backend = IvfIndex.load(native) if mode == "ivf" else HnswIndex.load(native)

            # Default the offset for snapshots that don't have it
            self.config["offset"] = int(meta["offset"]) if "offset" in meta else len(self.ids)
        except Exception:
            self.close()
            raise

    def index(self, embeddings):
        self.close()

        # Mode is validated before any state is stored
        mode = self.mode()

        self.directory = tempfile.mkdtemp()

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
        if not ids:
            return

        # Keep rows whose external id is not deleted. Unknown ids match nothing and are silently ignored.
        doomed = set(ids)
        keep = np.array([x not in doomed for x in self.ids.tolist()], dtype=bool)
        if keep.all():
            return

        self.vectors = np.ascontiguousarray(self.vectors[keep], dtype=np.float32)
        self.ids = self.ids[keep]

        self.build()

    def search(self, queries, limit):
        count = self.count()
        if not count:
            return [[] for _ in queries]

        queries = np.ascontiguousarray(queries, dtype=np.float32)

        # Clamp k to the live row count as the native index errors when k exceeds it
        k = min(limit, count)

        mode = self.mode()
        if mode == "ivf":
            nprobe = self.setting("nprobe", max(1, round(self.numclusters / 16)))
            ids, distances = self.backend.search(queries, k, nprobe)
        else:
            ef = self.setting("efsearch", None)
            ids, distances = self.backend.search(queries, k, ef or 0)

        # Map results to [(id, score)], dropping unfilled sentinel slots
        results = []
        for pids, dists in zip(ids.tolist(), distances.tolist()):
            results.append([(int(self.ids[pid]), float(1.0 - dist)) for pid, dist in zip(pids, dists) if pid != SENTINEL])

        return results

    def count(self):
        return len(self.ids) if self.ids is not None else 0

    def save(self, path):
        native = os.path.join(self.directory, "index.bin")
        if self.backend is not None:
            self.backend.save(native)
            self.backend = None
        elif os.path.exists(native):
            # Empty index: drop the stale native file so reloads don't resurrect deleted rows
            os.remove(native)

        np.savez(
            os.path.join(self.directory, "meta.npz"),
            vectors=self.vectors,
            ids=self.ids,
            mode=self.mode(),
            nbits=self.setting("nbits", 1),
            clusters=self.numclusters,
            nprobe=self.setting("nprobe", max(1, round(self.numclusters / 16))),
            m=self.setting("m", 16),
            efconstruction=self.setting("efconstruction", 200),
            efsearch=self.setting("efsearch", None) or -1,
            randomseed=self.setting("randomseed", 100),
            offset=self.config.get("offset", len(self.ids)),
        )

        archive = ArchiveFactory.create(self.directory)
        archive.save(path, "tar")

        # Reopen the native index after archiving
        if os.path.exists(native):
            mode = self.mode()
            self.backend = IvfIndex.load(native) if mode == "ivf" else HnswIndex.load(native)

    def close(self):
        # Parent logic releases the backend
        super().close()

        if self.directory and os.path.exists(self.directory):
            shutil.rmtree(self.directory)

        self.directory = None
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
        Builds the native index from the retained corpus.
        """

        rows = self.vectors.shape[0]
        if not rows:
            self.backend = None
            self.numclusters = 0
            return

        # pylint: disable=C0415
        from sklearn.cluster import KMeans

        mode = self.mode()
        dim = self.config["dimensions"]

        # Quantization bits per dimension, validated by rabitqlib
        nbits = self.setting("nbits", 1)

        # Default clusters to 4 * sqrt(N), bounded to [1, N] so KMeans always fits
        clusters = self.setting("clusters", None)
        self.numclusters = max(1, min(clusters if clusters else round(4 * math.sqrt(rows)), rows))

        cluster = KMeans(n_clusters=self.numclusters, random_state=0, n_init=10).fit(self.vectors)
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
