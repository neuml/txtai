import warnings
from txtai.ann import ANN

try:
    from grpc import RpcError
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.http.models import (
        PointIdsList,
        VectorParams,
        Distance,
        SearchRequest,
        SearchParams,
    )

    QDRANT_INSTALLED = True
except ImportError:
    QDRANT_INSTALLED = False


class Qdrant(ANN):
    """
    ANN implementation using Qdrant - https://qdrant.tech as a backend.
    """

    DISTANCE_MAPPING = {
        "cosine": Distance.COSINE,
        "l2": Distance.EUCLID,
        "ip": Distance.DOT,
        "l1": Distance.MANHATTAN,
    }

    def __init__(self, config):
        super().__init__(config)

        if not QDRANT_INSTALLED:
            raise ImportError("'qdrant_client' is not installed. " "Install txtai with the 'ann' extra to enable Qdrant ANN backend.")
        self.qdrant_config = self.config.get("qdrant", {})
        self.collection_name = self.qdrant_config.get("collection", "txtai-embeddings")
        self.qdrant_client = QdrantClient(
            location=self.qdrant_config.get("location"),
            url=self.qdrant_config.get("url"),
            port=self.qdrant_config.get("port", 6333),
            grpc_port=self.qdrant_config.get("grpc_port", 6334),
            prefer_grpc=self.qdrant_config.get("prefer_grpc", False),
            https=self.qdrant_config.get("https"),
            api_key=self.qdrant_config.get("api_key"),
            prefix=self.qdrant_config.get("prefix"),
            timeout=self.qdrant_config.get("timeout"),
            host=self.qdrant_config.get("host"),
            path=self.qdrant_config.get("path"),
            grpc_options=self.qdrant_config.get("grpc_options"),
        )

        # Initial offset is set to the number of existing rows
        try:
            self.config["offset"] = self.count()
        except (UnexpectedResponse, RpcError, ValueError):
            self.config["offset"] = 0

    def index(self, embeddings):
        vector_size = self.config.get("dimensions")
        metric_name = self.config.get("metric", "cosine")
        if metric_name not in self.DISTANCE_MAPPING:
            raise ValueError(f"Unsupported Qdrant similarity metric: {metric_name}")
        collection_config = self.qdrant_config.get("collection_config", {})

        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=self.DISTANCE_MAPPING[metric_name],
            ),
            **collection_config,
        )

        self.config["offset"] = 0
        self.append(embeddings)

    def append(self, embeddings):
        offset = self.config.get("offset", 0)
        new_count = embeddings.shape[0]
        ids = list(range(offset, offset + new_count))
        self.qdrant_client.upload_collection(
            collection_name=self.collection_name,
            vectors=embeddings,
            ids=ids,
        )
        self.config["offset"] += new_count

    def delete(self, ids):
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )

    def search(self, queries, limit):
        search_params = self.qdrant_config.get("search_params", {})
        search_results = self.qdrant_client.search_batch(
            collection_name=self.collection_name,
            requests=[
                SearchRequest(
                    vector=query.tolist(),
                    params=SearchParams(**search_params),
                    limit=limit,
                )
                for query in queries
            ],
        )

        results = []
        for search_result in search_results:
            results.append([(entry.id, entry.score) for entry in search_result])
        return results

    def count(self):
        result = self.qdrant_client.count(
            collection_name=self.collection_name,
        )
        return result.count

    def load(self, path):
        warnings.warn(
            "Trying to call .load method on Qdrant ANN backend. " "This is redundant and won't have any effect.",
            UserWarning,
        )

    def save(self, path):
        warnings.warn(
            "Trying to call .save method on Qdrant ANN backend. " "This is redundant and won't have any effect.",
            UserWarning,
        )
