"""
Hugging Face module
"""

from ...models import Models, PoolingFactory

from ..base import Vectors


class HFVectors(Vectors):
    """
    Builds vectors using the Hugging Face transformers library.
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

        return method in ("transformers", "pooling", "clspooling", "maxpooling", "meanpooling")

    def loadmodel(self, path):
        # Build embeddings with transformers pooling
        return PoolingFactory.create(
            {
                "method": self.config.get("method"),
                "path": path,
                "device": Models.deviceid(self.config.get("gpu", True)),
                "tokenizer": self.config.get("tokenizer"),
                "maxlength": self.config.get("maxlength"),
                "loadprompts": "instructions" not in self.config,
                "modelargs": self.config.get("vectors", {}),
            }
        )

    def encode(self, data, category=None):
        # Encode data using vectors model
        embeddings = self.model.encode(data, batch=self.encodebatch, category=category)

        # Multi-vector outputs can't be indexed directly, a fixed dimensional encoder is required
        if embeddings.ndim == 3:
            raise ValueError("late interaction models require a fixed dimensional encoder (muvera or lemur) to produce embeddings vectors")

        return embeddings
