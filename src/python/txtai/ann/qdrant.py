"""
Qdrant module
"""

# Conditional import
try:
    # pylint: disable=E0611
    from qdrant_client import QdrantClient
    from qdrant_client.http.models.models import Distance, PointIdsList

    QDRANT = True
except ImportError:
    QDRANT = False

from .base import ANN

DEFAULT_COLLECTION_NAME = "embeddings"


class Qdrant(ANN):
    distance_mapping = {
        "ip": Distance.DOT,
        "l2": Distance.EUCLID,
        "cosine": Distance.COSINE,
    }

    @property
    def _collection_name(self):
        return self.config.get("qdrant", {}).get("collection_name", DEFAULT_COLLECTION_NAME)

    @property
    def _distance(self):
        return self.distance_mapping.get(self.config.get("metric"), Distance.COSINE)

    def _connect(self):
        self.model = QdrantClient(
            **self.config.get("qdrant", {}).get("connection", {})
        )

    def load(self, _path):
        self._connect()

    def index(self, embeddings):
        self._connect()
        self.model.recreate_collection(
            collection_name=self._collection_name,
            vector_size=self.config["dimensions"],
            distance=self._distance,
            **self.config.get("qdrant", {}).get("collection", {})
        )
        self.model.upload_collection(
            collection_name=self._collection_name,
            vectors=embeddings,
            **self.config.get("qdrant", {}).get("upload", {})
        )

        self.config["offset"] = embeddings.shape[0]

    def append(self, embeddings):
        new = embeddings.shape[0]
        self.model.upload_collection(
            collection_name=self._collection_name,
            vectors=embeddings,
            ids=range(self.config["offset"], self.config["offset"] + new)
        )

        self.config["offset"] += new

    def delete(self, ids):
        self.model.delete(
            collection_name=self._collection_name,
            points_selector=PointIdsList(points=ids),
        )

    def search(self, queries, limit):
        results = []
        for query in queries:
            hits = self.model.search(
                collection_name=self._collection_name,
                query_vector=query.tolist(),
                search_params=self.config.get("qdrant", {}).get("search_params", {}),
                limit=limit,
            )
            results.append([(hit.id, hit.score) for hit in hits])
        return results

    def count(self):
        return self.model.count(collection_name=self._collection_name).count

    def save(self, _path):
        pass
