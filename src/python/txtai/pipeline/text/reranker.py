"""
Reranker module
"""

import hashlib
import json
import logging
from collections import OrderedDict
from threading import RLock

from ..base import Pipeline
from ...util import Library

library = Library()
np, torch, safetensors = library.numpy(), library.torch(), library.safetensors()
logger = logging.getLogger(__name__)


class Reranker(Pipeline):
    """
    Runs embeddings queries and re-ranks them using a similarity pipeline. Note that content must be enabled with the
    embeddings instance for this to work properly.
    """

    # Cache lock shared by all instances, since rerankers can share one similarity pipeline
    lock = RLock()

    def __init__(self, embeddings, similarity, cache=None):
        """
        Creates a Reranker pipeline.

        Args:
            embeddings: embeddings instance (content must be enabled)
            similarity: similarity instance
            cache: optional document token cache; True bounds at 1000, False/None disable; other values use int(cache), non-positive disables
        """

        self.embeddings, self.similarity = embeddings, similarity
        self.cachesize = 1000 if cache is True else 0 if cache is None or cache is False else int(cache)
        self.cache = OrderedDict() if self.cachesize > 0 else None
        self.namespace = None

    # pylint: disable=W0222
    def __call__(self, query, limit=3, factor=10, **kwargs):
        """
        Runs an embeddings search and re-ranks the results using a Similarity pipeline.

        Args:
            query: query text|list
            limit: maximum results
            factor: factor to multiply limit by for the initial embeddings search
            kwargs: additional arguments to pass to embeddings search

        Returns:
            list of query results rescored using a Similarity pipeline
        """

        queries = [query] if not isinstance(query, list) else query

        # Run searches
        results = self.embeddings.batchsearch(queries, limit * factor, **kwargs)

        # Re-rank using similarity pipeline
        ranked = []
        for x, result in enumerate(results):
            # Skip scoring when a search has no results
            if not result:
                ranked.append([])
                continue

            texts = [row["text"] for row in result]

            if self.cache is not None and (namespace := self.modelkey()) and all("id" in row for row in result):
                with self.lock:
                    texts = self.vectors(result, namespace)
                    for uid, score in self.similarity(queries[x], texts):
                        result[uid]["score"] = score
            else:
                # Score results and merge
                for uid, score in self.similarity(queries[x], texts):
                    result[uid]["score"] = score

            # Sort and take top n sorted results
            ranked.append(sorted(result, key=lambda row: row["score"], reverse=True)[:limit])

        return ranked[0] if isinstance(query, str) else ranked

    def modelkey(self):
        """Compute a stable namespace for batch-independent document encoding."""

        encoder = getattr(self.similarity, "lateencoder", None)
        if not encoder:
            return None

        pooling = encoder.model
        center = getattr(pooling, "center", None)
        if center and center["scope"] == "batch":
            return None

        mean = center.get("mean") if center else None
        identity = {
            "pooling": type(pooling).__name__,
            "model": getattr(pooling.model.config, "_name_or_path"),
            "revision": getattr(pooling.model.config, "_commit_hash", None),
            "tokenizer": pooling.tokenizer.name_or_path,
            "scope": center["scope"] if center else None,
            "mean": hashlib.sha256(np.asarray(mean).tobytes()).hexdigest() if mean is not None else None,
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()

    def reset(self, namespace):
        """Clear entries when the encoder namespace changes; caller holds the cache lock."""

        if self.namespace != namespace:
            self.cache.clear()
            self.namespace = namespace

    def vectors(self, rows, namespace):
        """Gather cached vectors and encode misses once, preserving candidate positions."""

        with self.lock:
            self.reset(namespace)
            keys, vectors, misses = [], [], []
            for row in rows:
                key = (namespace, row["id"], hashlib.sha256(row["text"].encode("utf-8")).hexdigest())
                vector = self.cache.get(key)
                keys.append(key)
                vectors.append(vector)
                if vector is None:
                    misses.append(len(vectors) - 1)
                else:
                    self.cache.move_to_end(key)

            if misses:
                encoded = self.similarity.encode([rows[x]["text"] for x in misses], "data")
                for position, vector in zip(misses, encoded):
                    vectors[position] = vector[vector.abs().sum(dim=-1) > 0].detach().to(device="cpu", dtype=torch.float32)

            # Keep local references through padding before LRU insertion can evict any current candidate.
            batch = torch.zeros((len(vectors), max(vector.shape[0] for vector in vectors), vectors[0].shape[1]), dtype=torch.float32)
            for position, vector in enumerate(vectors):
                batch[position, : vector.shape[0]] = vector

            for position in misses:
                self.cache.update({keys[position]: vectors[position]})
                while len(self.cache) > self.cachesize:
                    self.cache.popitem(last=False)

            return batch.to(self.similarity.lateencoder.device)

    def save(self, path):
        """Save an independent safetensors snapshot; disabled or unsupported caches do nothing."""

        if self.cache is not None and (namespace := self.modelkey()):
            with self.lock:
                self.reset(namespace)
                metadata = {"version": "1", "namespace": namespace, "keys": json.dumps([list(key[1:]) for key in self.cache.keys()])}
                safetensors.numpy.save_file({str(x): vector.numpy() for x, vector in enumerate(self.cache.values())}, path, metadata)

    def load(self, path):
        """Load a bounded snapshot, warning on model mismatch; disabled or unsupported caches do nothing."""

        if self.cache is not None and (namespace := self.modelkey()):
            with self.lock:
                self.cache.clear()
                self.namespace = namespace
                with safetensors.safe_open(path, framework="np") as source:
                    metadata = source.metadata()
                    if metadata["namespace"] != namespace:
                        logger.warning("Ignoring reranker cache snapshot with a different encoder namespace")
                        return
                    for x, key in enumerate(json.loads(metadata["keys"])):
                        self.cache.update({(namespace, *key): torch.from_numpy(source.get_tensor(str(x)))})
                        while len(self.cache) > self.cachesize:
                            self.cache.popitem(last=False)
