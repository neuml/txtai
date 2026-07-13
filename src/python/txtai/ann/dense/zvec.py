"""
zvec module
"""

import os
import shutil
import tarfile
import tempfile

# Conditional import
try:
    import zvec

    ZVEC = True
except ImportError:
    ZVEC = False

from ..base import ANN


class Zvec(ANN):
    """
    Builds an ANN index using the zvec library.
    """

    def __init__(self, config):
        super().__init__(config)

        if not ZVEC:
            raise ImportError('zvec is not available - install "ann" extra to enable')

        self.directory = None
        self.path = None

    def load(self, path):
        self.close()
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "index")

        try:
            with tarfile.open(path, "r") as archive:
                members = archive.getmembers()

                for member in members:
                    target = os.path.abspath(os.path.join(self.directory, member.name))
                    if os.path.commonpath([self.directory, target]) != self.directory or member.issym() or member.islnk():
                        raise ValueError("Invalid zvec index archive")

                archive.extractall(self.directory, members=members)

            self.backend = zvec.open(self.path)
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
        schema = zvec.CollectionSchema(
            name="txtai",
            vectors=zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                self.config["dimensions"],
                index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.IP, m=m),
            ),
        )
        self.backend = zvec.create_and_open(path=self.path, schema=schema)

        # Add items - position in embeddings is used as the id
        self.insert(embeddings, 0)

        # Add id offset and index build metadata
        self.config["offset"] = embeddings.shape[0]
        self.metadata({"m": m, "zvec": zvec.__version__})

    def append(self, embeddings):
        self.insert(embeddings, self.config["offset"])

        # Update id offset and index metadata
        self.config["offset"] += embeddings.shape[0]
        self.metadata()

    def delete(self, ids):
        if ids:
            self.backend.delete([str(uid) for uid in ids])

    def search(self, queries, limit):
        results = []
        for query in queries:
            matches = self.backend.query(
                zvec.Query(field_name="embedding", vector=query.tolist()),
                topk=limit,
            )
            results.append([(int(match.id), float(match.score)) for match in matches])

        return results

    def count(self):
        return self.backend.stats.doc_count

    def save(self, path):
        self.backend.flush()

        # Write a single-file artifact that contains the directory-shaped collection
        descriptor, temporary = tempfile.mkstemp(dir=os.path.dirname(path) or ".")
        os.close(descriptor)

        try:
            with tarfile.open(temporary, "w") as archive:
                archive.add(self.path, arcname="index")

            if os.path.isdir(path):
                shutil.rmtree(path)
            os.replace(temporary, path)
        except Exception:
            if os.path.exists(temporary):
                os.remove(temporary)
            raise

    def close(self):
        # Parent logic releases the collection and its file locks
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
                self.backend.insert(
                    [zvec.Doc(id=str(offset + start + uid), vectors={"embedding": embedding.tolist()}) for uid, embedding in enumerate(batch)]
                )
