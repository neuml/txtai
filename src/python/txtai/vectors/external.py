"""
External module
"""

import pickle
import tempfile

import numpy as np

from .base import Vectors


class ExternalVectors(Vectors):
    """
    Loads pre-computed vectors. Pre-computed vectors allow integrating other vector models. They can also be used to efficiently
    test different model configurations without having to recompute vectors.
    """

    def load(self, path):
        return None

    def index(self, documents):
        ids, dimensions, batches, stream = [], None, 0, None

        # Convert all documents to embedding arrays, stream embeddings to disk to control memory usage
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy", delete=False) as output:
            stream = output.name
            batch = []
            for document in documents:
                # Require embeddings vector as input
                if isinstance(document[1], np.ndarray):
                    batch.append(document)

                if len(batch) == 500:
                    # Convert batch to embeddings
                    uids, dimensions = self.batch(batch, output)
                    ids.extend(uids)
                    batches += 1

                    batch = []

            # Final batch
            if batch:
                uids, dimensions = self.batch(batch, output)
                ids.extend(uids)
                batches += 1

        return (ids, dimensions, batches, stream)

    def transform(self, document):
        # Get transform function
        transform = self.config.get("transform")

        # Apply transform function if set, otherwise, assume input is an array
        return transform(document) if transform else document[1].astype(np.float32)

    def batch(self, documents, output):
        """
        Builds a batch of embeddings.

        Args:
            documents: list of documents used to build embeddings
            output: output temp file to store embeddings

        Returns:
            (ids, dimensions) list of ids and number of dimensions in embeddings
        """

        # Extract ids and prepare input documents for transformers model
        ids = [uid for uid, _, _ in documents]
        dimensions = None

        # Build embeddings
        embeddings = np.array([data for _, data, _ in documents])
        if embeddings is not None:
            dimensions = embeddings.shape[1]
            pickle.dump(embeddings, output, protocol=4)

        return (ids, dimensions)
