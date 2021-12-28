"""
Transformers module
"""

import os
import pickle
import tempfile

# Conditional import
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS = True
except ImportError:
    SENTENCE_TRANSFORMERS = False

from .base import Vectors
from ..models import MeanPooling, Models, Pooling
from ..pipeline import Tokenizer


class TransformersVectors(Vectors):
    """
    Builds sentence embeddings/vectors using the transformers library.
    """

    def load(self, path):
        # Flag that determines if transformers or sentence-transformers should be used to build embeddings
        method = self.config.get("method")
        transformers = method != "sentence-transformers"

        # Tensor device id
        deviceid = Models.deviceid(self.config.get("gpu", True))

        # Build embeddings with transformers (default)
        if transformers:
            if isinstance(path, bytes) or (isinstance(path, str) and os.path.isfile(path)) or method == "pooling":
                return Pooling(path, device=deviceid, tokenizer=self.config.get("tokenizer"))

            return MeanPooling(path, device=deviceid, tokenizer=self.config.get("tokenizer"))

        if not SENTENCE_TRANSFORMERS:
            raise ImportError('sentence-transformers is not available - install "similarity" extra to enable')

        # Build embeddings with sentence-transformers
        return SentenceTransformer(path, device=Models.reference(deviceid))

    def index(self, documents):
        ids, dimensions, batches, stream = [], None, 0, None

        # Convert all documents to embedding arrays, stream embeddings to disk to control memory usage
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy", delete=False) as output:
            stream = output.name
            batch = []
            for document in documents:
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
        # Prepare input document for transformers model and build embeddings
        return self.model.encode([self.prepare(document[1])])[0]

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
        documents = [self.prepare(data) for _, data, _ in documents]
        dimensions = None

        # Build embeddings
        embeddings = self.model.encode(documents)
        if embeddings is not None:
            dimensions = embeddings.shape[1]
            pickle.dump(embeddings, output, protocol=4)

        return (ids, dimensions)

    def prepare(self, data):
        """
        Prepares input data for transformers model. This method supports optional string tokenization and
        joins tokenized input into text.

        Args:
            data: input data

        Returns:
            data formatted for transformers model
        """

        # Optional string tokenization
        if self.tokenize and isinstance(data, str):
            data = Tokenizer.tokenize(data)

        # Convert token list to string
        if isinstance(data, list):
            data = " ".join(data)

        return data
