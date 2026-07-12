"""
Milvus module
"""

import os
import shutil
import tempfile

from importlib.metadata import PackageNotFoundError, version

# Conditional import
try:
    from pymilvus import DataType, MilvusClient

    MILVUS = True
except ImportError:
    MILVUS = False

from ..base import ANN

# Core library imports
from ...util import Library

np = Library().numpy()


class Milvus(ANN):
    """
    Builds an ANN index using Milvus Lite.
    """

    COLLECTION = "txtai"
    PRIMARY = "indexid"
    VECTOR = "embedding"
    METRIC = "IP"

    def __init__(self, config):
        super().__init__(config)

        if not MILVUS:
            raise ImportError('Milvus is not available - install "ann" extra to enable')

        self.directory = None
        self.path = None

    def load(self, path):
        self.close()
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "milvus.db")

        try:
            self.copy(path, self.path)
            self.initialize()
        except Exception:
            self.close()
            raise

    def index(self, embeddings):
        self.close()

        # Initialize collection
        self.initialize(recreate=True)

        # Add embeddings - position in embeddings is used as the id
        self.insert(embeddings, 0)

        # Add id offset and index build metadata
        self.config["offset"] = embeddings.shape[0]
        self.metadata(self.settings())

    def append(self, embeddings):
        # Append new ids - position in embeddings + existing offset is used as the id
        self.insert(embeddings, self.config["offset"])

        # Update id offset and index metadata
        self.config["offset"] += embeddings.shape[0]
        self.metadata()

    def delete(self, ids):
        # Delete specified ids
        if ids:
            self.database().delete(collection_name=self.COLLECTION, ids=[int(uid) for uid in ids])

    def search(self, queries, limit):
        # Run the query
        results = self.database().search(
            collection_name=self.COLLECTION,
            data=np.asarray(queries, dtype=np.float32).tolist(),
            anns_field=self.VECTOR,
            limit=limit,
            output_fields=[self.PRIMARY],
            search_params={"metric_type": self.METRIC},
        )

        # Map results to [(id, score)]
        return [[(self.uid(result), self.score(result)) for result in query] for query in results]

    def count(self):
        stats = self.database().get_collection_stats(collection_name=self.COLLECTION)
        return int(stats["row_count"])

    def save(self, path):
        # Flush pending writes
        if self.backend:
            self.backend.flush(collection_name=self.COLLECTION)
            self.release()

        # Copy the current Milvus Lite database to the target path
        if self.path and path:
            if os.path.abspath(self.path) != os.path.abspath(path):
                self.copy(self.path, path)

            self.connect()

    def close(self):
        # Parent logic releases the client
        self.release()
        super().close()

        if self.directory and os.path.exists(self.directory):
            shutil.rmtree(self.directory)

        self.directory = None
        self.path = None

    def initialize(self, recreate=False):
        """
        Initializes a Milvus collection.

        Args:
            recreate: Recreates the collection if True
        """

        # Connect to Milvus
        self.connect()

        # Drop collection, if necessary
        if recreate and self.backend.has_collection(collection_name=self.COLLECTION):
            self.backend.drop_collection(collection_name=self.COLLECTION)

        # Create collection, if necessary
        if not self.backend.has_collection(collection_name=self.COLLECTION):
            self.create()
        else:
            self.validate()

    def connect(self):
        """
        Establishes a Milvus client connection.
        """

        # Reuse current connection
        if self.backend:
            return

        if not self.path:
            self.directory = tempfile.mkdtemp()
            self.path = os.path.join(self.directory, "milvus.db")

        self.backend = MilvusClient(uri=self.path)

    def create(self):
        """
        Creates the Milvus collection.
        """

        # Define schema
        schema = self.backend.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name=self.PRIMARY, datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name=self.VECTOR, datatype=DataType.FLOAT_VECTOR, dim=self.config["dimensions"])

        # Create vector index
        index = self.backend.prepare_index_params()
        index.add_index(field_name=self.VECTOR, index_type="AUTOINDEX", metric_type=self.METRIC)

        # Create collection
        self.backend.create_collection(
            collection_name=self.COLLECTION, schema=schema, index_params=index,
        )

    def validate(self):
        """
        Validates the Milvus collection schema.
        """

        fields = {field["name"]: field for field in self.backend.describe_collection(collection_name=self.COLLECTION)["fields"]}

        if self.PRIMARY not in fields:
            raise ValueError(f"Milvus collection '{self.COLLECTION}' is missing '{self.PRIMARY}' primary key field")

        if self.VECTOR not in fields:
            raise ValueError(f"Milvus collection '{self.COLLECTION}' is missing '{self.VECTOR}' vector field")

        primary = fields[self.PRIMARY]
        if primary["type"] != DataType.INT64 or not primary.get("is_primary"):
            raise ValueError(f"Milvus collection '{self.COLLECTION}' must use an INT64 '{self.PRIMARY}' primary key")

        vector = fields[self.VECTOR]
        if vector["type"] != DataType.FLOAT_VECTOR or int(vector["params"]["dim"]) != self.config["dimensions"]:
            raise ValueError(f"Milvus collection '{self.COLLECTION}' must use a FLOAT_VECTOR '{self.VECTOR}' field with configured dimensions")

    def insert(self, embeddings, offset):
        """
        Inserts embeddings into Milvus.

        Args:
            embeddings: embeddings array
            offset: id offset
        """

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.shape[0]:
            self.database().insert(
                collection_name=self.COLLECTION,
                data=[{self.PRIMARY: int(x + offset), self.VECTOR: row.tolist()} for x, row in enumerate(embeddings)],
            )

    def database(self):
        """
        Gets the current Milvus client. Creates a new connection if there isn't one.

        Returns:
            Milvus client
        """

        if not self.backend:
            self.initialize()

        return self.backend

    def copy(self, source, target):
        """
        Copies a Milvus Lite database.

        Args:
            source: source database path
            target: target database path
        """

        if os.path.exists(target):
            if os.path.isdir(target):
                shutil.rmtree(target)
            else:
                os.remove(target)

        if os.path.isdir(source):
            shutil.copytree(source, target)
        else:
            shutil.copyfile(source, target)

    def settings(self):
        """
        Returns settings for this index.

        Returns:
            dict
        """

        settings = {"milvus": "lite", "metric_type": self.METRIC}

        try:
            settings["pymilvus"] = version("pymilvus")
        except PackageNotFoundError:
            pass

        try:
            settings["milvus-lite"] = version("milvus-lite")
        except PackageNotFoundError:
            pass

        return settings

    def uid(self, result):
        """
        Extracts an id from a Milvus search result.

        Args:
            result: Milvus search result

        Returns:
            result id
        """

        return int(result.get(self.PRIMARY, result.get("id", result.get("entity", {}).get(self.PRIMARY))))

    def score(self, result):
        """
        Extracts a score from a Milvus search result.

        Args:
            result: Milvus search result

        Returns:
            score
        """

        score = result.get("distance", result.get("score"))
        return score

    def release(self):
        """
        Releases the current Milvus client.
        """

        if self.backend:
            self.backend.close()
            self.backend = None
