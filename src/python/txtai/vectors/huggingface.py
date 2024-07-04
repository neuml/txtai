"""
Hugging Face module
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


class HFVectors(Vectors):
    """
    Builds vectors using the Hugging Face transformers library. Also supports the sentence-transformers library.
    """

    @staticmethod
    def ismethod(method):
        """
        Checks if this method uses local transformers-based models.

        Args:
            method: input method

        Returns:
            True if this is a local transformers-based model, False otherwise
        """

        return method in ("transformers", "sentence-transformers", "pooling", "clspooling", "meanpooling")

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
                {
                    "method": method,
                    "path": path,
                    "device": deviceid,
                    "tokenizer": self.config.get("tokenizer"),
                    "maxlength": self.config.get("maxlength"),
                    "modelargs": modelargs,
                }
            )

        # Otherwise, use sentence-transformers library
        if not SENTENCE_TRANSFORMERS:
            raise ImportError('sentence-transformers is not available - install "vectors" extra to enable')

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
