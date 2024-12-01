"""
SentenceTransformers module
"""

# Conditional import
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS = True
except ImportError:
    SENTENCE_TRANSFORMERS = False

from ..models import Models

from .base import Vectors


class STVectors(Vectors):
    """
    Builds vectors using sentence-transformers (aka SBERT).
    """

    def __init__(self, config, scoring, models):
        # Check before parent constructor since it calls loadmodel
        if not SENTENCE_TRANSFORMERS:
            raise ImportError('sentence-transformers is not available - install "vectors" extra to enable')

        super().__init__(config, scoring, models)

    def loadmodel(self, path):
        # Tensor device id
        deviceid = Models.deviceid(self.config.get("gpu", True))

        # Additional model arguments
        modelargs = self.config.get("vectors", {})

        # Build embeddings with sentence-transformers
        return SentenceTransformer(path, device=Models.device(deviceid), **modelargs)

    def encode(self, data):
        # Encode data using vectors model
        return self.model.encode(data, batch_size=self.encodebatch)
