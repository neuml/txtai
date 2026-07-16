"""
milvus-lite module
"""

import os
import shutil
import tempfile

# Conditional import
try:
    import milvus_lite

    MILVUSLITE = True
except ImportError:
    MILVUSLITE = False

from ...archive import ArchiveFactory
from ..base import ANN


class MilvusLite(ANN):
    """
    Builds an ANN index using the milvus-lite library.
    """

    def __init__(self, config):
        super().__init__(config)

        if not MILVUSLITE:
            raise ImportError('milvus-lite is not available - install "ann" extra to enable')

        self.collection = None
        self.directory = None
        self.path = None

    def load(self, path):
        self.close()
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "index")

        try:
            archive = ArchiveFactory.create(self.directory)
            archive.load(path, "tar")
            self.open()
        except Exception:
            self.close()
            raise

    def index(self, embeddings):
        self.close()

        # Lookup index settings
        m = self.setting("m", 50)

        # Create collection
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "index")

        # Use milvus-lite's native embedded API directly and keep this integration thin
        self.backend = milvus_lite.MilvusLite(self.path)
        schema = milvus_lite.CollectionSchema(
            fields=[
                milvus_lite.FieldSchema("id", milvus_lite.DataType.INT64, is_primary=True),
                milvus_lite.FieldSchema("embedding", milvus_lite.DataType.FLOAT_VECTOR, dim=self.config["dimensions"]),
            ]
        )
        self.collection = self.backend.create_collection("txtai", schema)
        self.collection.create_index("embedding", {"index_type": "HNSW", "metric_type": "IP", "params": {"M": m}})
        self.collection.load()

        # Add items - position in embeddings is used as the id
        self.insert(embeddings, 0)

        # Add id offset and index build metadata
        self.config["offset"] = embeddings.shape[0]
        self.metadata({"m": m, "milvuslite": milvus_lite.__version__})

    def append(self, embeddings):
        self.insert(embeddings, self.config["offset"])

        # Update id offset and index metadata
        self.config["offset"] += embeddings.shape[0]
        self.metadata()

    def delete(self, ids):
        if ids:
            self.collection.delete(ids)

    def search(self, queries, limit):
        matches = self.collection.search(queries.tolist(), top_k=limit, metric_type="IP", anns_field="embedding")
        return [[(int(match["id"]), float(match["distance"])) for match in results] for results in matches]

    def count(self):
        return self.collection.num_entities

    def save(self, path):
        self.collection.flush()
        self.backend.close()
        self.collection = None
        self.backend = None
        archive = ArchiveFactory.create(self.directory)
        archive.save(path, "tar")
        self.open()

    def close(self):
        # Release milvus-lite collection handles and the directory lock
        if self.backend:
            self.backend.close()
        self.collection = None
        super().close()

        if self.directory and os.path.exists(self.directory):
            shutil.rmtree(self.directory)

        self.directory = None
        self.path = None

    def insert(self, embeddings, offset):
        """
        Inserts embeddings starting at offset.

        Args:
            embeddings: embeddings array
            offset: starting id
        """

        if embeddings.shape[0]:
            for start in range(0, embeddings.shape[0], 1024):
                batch = embeddings[start : start + 1024]
                self.collection.insert(
                    [{"id": offset + start + uid, "embedding": embedding.tolist()} for uid, embedding in enumerate(batch)]
                )

    def open(self):
        """Opens the milvus-lite database and collection."""

        self.backend = milvus_lite.MilvusLite(self.path)
        self.collection = self.backend.get_collection("txtai")
