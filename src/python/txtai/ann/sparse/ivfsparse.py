"""
IVFSparse module
"""

import math

import numpy as np

# Conditional import
try:
    from scipy.sparse import vstack
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    from sklearn.utils.extmath import safe_sparse_dot

    IVFSPARSE = True
except ImportError:
    IVFSPARSE = False

from ...serialize import SerializeFactory
from ...util import SparseArray
from ..base import ANN


class IVFSparse(ANN):
    """
    Inverted file (IVF) index with flat vector file storage and sparse array support.

    IVFSparse builds an IVF index and enables approximate nearest neighbor (ANN) search.

    This index is modeled after Faiss and supports many of the same parameters.

    See this link for more: https://github.com/facebookresearch/faiss/wiki/Faster-search
    """

    def __init__(self, config):
        super().__init__(config)

        if not IVFSPARSE:
            raise ImportError('IVFSparse is not available - install "ann" extra to enable')

        # Cluster centroids, if computed
        self.centroids = None

        # Cluster id mapping
        self.ids = None

        # Cluster data blocks - can be a single block with no computed centroids
        self.blocks = None

        # Deleted ids
        self.deletes = None

    def index(self, embeddings):
        # Compute model training size
        train, sample = embeddings, self.setting("sample")
        if sample:
            # Get sample for training
            rng = np.random.default_rng(0)
            indices = sorted(rng.choice(train.shape[0], int(sample * train.shape[0]), replace=False, shuffle=False))
            train = train[indices]

        # Get number of clusters. Note that final number of clusters could be lower due to filtering duplicate centroids.
        clusters = self.nlist(embeddings.shape[0], train.shape[0])

        # A single cluster is an exact search
        if clusters > 1:
            # Calculate number of data points
            kmeans = MiniBatchKMeans(n_clusters=clusters, random_state=0, n_init="auto").fit(train)

            # Find closest points to each cluster center and use those as centroids
            # Filter out duplicate centroids
            indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, train, metric="cosine")
            self.centroids = embeddings[np.unique(indices)]

        # Sort into clusters
        ids = self.aggregate(embeddings)

        # Sort clusters by id
        self.ids = dict(sorted(ids.items(), key=lambda x: x[0]))

        # Create cluster data blocks
        self.blocks = {k: embeddings[v] for k, v in self.ids.items()}

        # Initialize deletes
        self.deletes = []

        # Add id offset and index build metadata
        self.config["offset"] = embeddings.shape[0]
        self.metadata({"clusters": clusters})

    def append(self, embeddings):
        # Get offset
        offset = self.size()

        # Sort into clusters and merge
        for cluster, ids in self.aggregate(embeddings).items():
            # Add new ids
            self.ids[cluster].extend([x + offset for x in ids])

            # Add new data
            self.blocks[cluster] = vstack([self.blocks[cluster], embeddings[ids]])

        # Update id offset and index metadata
        self.config["offset"] += embeddings.shape[0]
        self.metadata()

    def delete(self, ids):
        # Set index ids as deleted
        self.deletes.extend(ids)

    def search(self, queries, limit):
        results = []
        for query in queries:
            if self.centroids is not None:
                # Approximate search
                indices, _ = self.topn(query, self.centroids, self.nprobe())

                # Stack into single ids list
                ids = np.concatenate([self.ids[x] for x in indices if x in self.ids])

                # Stack data rows
                data = vstack([self.blocks[x] for x in indices if x in self.blocks])
            else:
                # Exact search
                ids, data = np.array(self.ids[0]), self.blocks[0]

            # Get deletes
            deletes = np.argwhere(np.isin(ids, self.deletes)).ravel()

            # Calculate similarity
            indices, scores = self.topn(query, data, limit, deletes)

            # Map data ids and return
            results.append(list(zip(ids[indices].tolist(), scores.tolist())))

        return results

    def count(self):
        return self.size() - len(self.deletes)

    def load(self, path):
        # Create streaming serializer and limit read size to a byte at a time to ensure
        # only msgpack data is consumed
        serializer = SerializeFactory.create("msgpack", streaming=True, read_size=1)

        with open(path, "rb") as f:
            # Read header
            unpacker = serializer.loadstream(f)
            header = next(unpacker)

            # Read cluster centroids, if available
            self.centroids = SparseArray().load(f) if header["centroids"] else None

            # Read cluster ids
            self.ids = dict(next(unpacker))

            # Read cluster data blocks
            self.blocks = {}
            for key in self.ids:
                self.blocks[key] = SparseArray().load(f)

            # Read deletes
            self.deletes = next(unpacker)

    def save(self, path):
        # IVFSparse storage format:
        #    - header msgpack
        #    - centroids sparse array (optional based on header parameters)
        #    - cluster ids msgpack
        #    - cluster data blocks list of sparse arrays
        #    - deletes msgpack

        # Create message pack serializer
        serializer = SerializeFactory.create("msgpack")

        with open(path, "wb") as f:
            # Write header
            serializer.savestream({"centroids": self.centroids is not None, "count": self.count(), "blocks": len(self.blocks)}, f)

            # Write cluster centroids, if available
            if self.centroids is not None:
                SparseArray().save(f, self.centroids)

            # Write cluster id mapping
            serializer.savestream(list(self.ids.items()), f)

            # Write cluster data blocks
            for block in self.blocks.values():
                SparseArray().save(f, block)

            # Write deletes
            serializer.savestream(self.deletes, f)

    def aggregate(self, data):
        """
        Aggregates input data array into clusters. This method sorts each data element into the
        cluster with the highest cosine similarity centroid.

        Args:
            data: input data

        Returns:
            {cluster, ids}
        """

        # Exact search when only a single cluster
        if self.centroids is None:
            return {0: list(range(data.shape[0]))}

        # Map data to closest centroids
        indices, _ = pairwise_distances_argmin_min(data, self.centroids, metric="cosine")

        # Sort into clusters
        ids = {}
        for x, cluster in enumerate(indices.tolist()):
            if cluster not in ids:
                ids[cluster] = []

            # Save id
            ids[cluster].append(x)

        return ids

    def topn(self, query, data, limit, deletes=None):
        """
        Gets the top n most similar data elements for query.

        Args:
            query: input query array
            data: data array
            limit: top n
            deletes: optional list of deletes to filter from results

        Returns:
            list of matching (indices, scores)
        """

        # Dot product similarity (assumes all data is normalized)
        scores = safe_sparse_dot(query, data.T, dense_output=True)

        # Clear deletes
        if deletes is not None:
            scores[:, deletes] = 0

        # Get top n matching indices and scores
        indices = np.argpartition(-scores, limit if limit < scores.shape[0] else scores.shape[0] - 1)[:, :limit]
        scores = np.clip(np.take_along_axis(scores, indices, axis=1), 0.0, 1.0)

        return indices[0], scores[0]

    def nlist(self, count, train):
        """
        Calculates the number of clusters for this IVFSparse index. Note that the final number of clusters
        could be lower as duplicate cluster centroids are filtered out.

        Args:
            count: initial dataset size
            train: number of rows used to train

        Returns:
            number of clusters
        """

        # Get data size
        default = 1 if count <= 5000 else self.cells(train)

        # Number of clusters to create
        return self.setting("nlist", default)

    def nprobe(self):
        """
        Gets or derives the nprobe search parameter.

        Returns:
            nprobe setting
        """

        # Get size of embeddings index
        size = self.size()

        default = 6 if size <= 5000 else self.cells(size) // 64
        return self.setting("nprobe", default)

    def cells(self, count):
        """
        Calculates the number of IVF cells for an IVFSparse index.

        Args:
            count: number of rows

        Returns:
            number of IVF cells
        """

        # Calculate number of IVF cells where x = min(4 * sqrt(count), count / 39)
        # Match faiss behavior that requires at least 39 * x data points
        return max(min(round(4 * math.sqrt(count)), int(count / 39)), 1)

    def size(self):
        """
        Gets the total size of this index including deletes.

        Returns:
            size
        """

        return sum(len(x) for x in self.ids.values())
