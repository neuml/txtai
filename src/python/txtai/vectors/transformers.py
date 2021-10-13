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
from ..pipeline.tokenizer import Tokenizer


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
        ids, dimensions, stream = [], None, None

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

                    batch = []

            # Final batch
            if batch:
                uids, dimensions = self.batch(batch, output)
                ids.extend(uids)

        return (ids, dimensions, stream)

    def transform(self, document):
        # Convert input document to text and build embeddings
        return self.model.encode([self.text(document[1])])[0]

    def batch(self, documents, output):
        """
        Builds a batch of embeddings.

        Args:
            documents: list of documents used to build embeddings
            output: output temp file to store embeddings

        Returns:
            (ids, dimensions) list of ids and number of dimensions in embeddings
        """

        # Extract ids and convert input documents to text
        ids = [uid for uid, _, _ in documents]
        documents = [self.text(text) for _, text, _ in documents]
        dimensions = None

        # Build embeddings
        embeddings = self.model.encode(documents)
        for embedding in embeddings:
            if not dimensions:
                # Set number of dimensions for embeddings
                dimensions = embedding.shape[0]

            pickle.dump(embedding, output)

        return (ids, dimensions)

    def text(self, text):
        """
        Converts input into text that can be processed by transformer models. This method supports
        optional string tokenization and joins tokenized input into text.

        Args:
            text: text|tokens
        """

        # Optional string tokenization
        if self.tokenize and isinstance(text, str):
            text = Tokenizer.tokenize(text)

        # Transformer models require string input
        if isinstance(text, list):
            text = " ".join(text)

        return text
