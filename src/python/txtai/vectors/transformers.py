"""
Transformers module
"""

# Conditional import
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS = True
except ImportError:
    SENTENCE_TRANSFORMERS = False

from .base import Vectors
from ..models import Models, PoolingFactory
from ..pipeline import Tokenizer


class TransformersVectors(Vectors):
    """
    Builds sentence embeddings/vectors using the transformers library.
    """

    def loadmodel(self, path):
        # Flag that determines if transformers or sentence-transformers should be used to build embeddings
        method = self.config.get("method")
        transformers = method != "sentence-transformers"

        # Tensor device id
        deviceid = Models.deviceid(self.config.get("gpu", True))

        # Additional model arguments
        modelargs = self.config.get("vectors", {})

        # Build embeddings with transformers (default)
        if transformers:
            return PoolingFactory.create(
                {"path": path, "device": deviceid, "tokenizer": self.config.get("tokenizer"), "method": method, "modelargs": modelargs}
            )

        # Otherwise, use sentence-transformers library
        if not SENTENCE_TRANSFORMERS:
            raise ImportError('sentence-transformers is not available - install "similarity" extra to enable')

        # Build embeddings with sentence-transformers
        return SentenceTransformer(path, device=Models.device(deviceid), **modelargs)

    def encode(self, data):
        # Get batch parameter name
        param = "batch_size" if self.config.get("method") == "sentence-transformers" else "batch"

        # Encode data using vectors model
        return self.model.encode(data, **{param: self.encodebatch})

    def prepare(self, data, category=None):
        # Optional string tokenization
        if self.tokenize and isinstance(data, str):
            data = Tokenizer.tokenize(data)

        # Convert token list to string
        if isinstance(data, list):
            data = " ".join(data)

        # Add parent prepare logic
        return super().prepare(data, category)
