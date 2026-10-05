"""
Sparse Sentence Transformers module
"""

# Conditional import
try:
    from sentence_transformers import SparseEncoder

    SENTENCE_TRANSFORMERS = True
except ImportError:
    SENTENCE_TRANSFORMERS = False

from ..dense.sbert import STVectors
from .base import SparseVectors


class SparseSTVectors(SparseVectors, STVectors):
    """
    Builds sparse vectors using sentence-transformers (aka SBERT).
    """

    def __init__(self, config, scoring, models):
        # Check before parent constructor since it calls loadmodel
        if not SENTENCE_TRANSFORMERS:
            raise ImportError('sentence-transformers is not available - install "vectors" extra to enable')

        super().__init__(config, scoring, models)

    def loadencoder(self, path, device, **kwargs):
        return SparseEncoder(path, device=device, **kwargs)

    def defaultnormalize(self):
        # Enable normalization by default if similarity function is cosine
        return self.model and self.model.similarity_fn_name == "cosine"
