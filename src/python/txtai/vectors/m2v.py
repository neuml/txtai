"""
Model2Vec module
"""

import json

from huggingface_hub.errors import HFValidationError
from transformers.utils import cached_file

# Conditional import
try:
    from model2vec import StaticModel

    MODEL2VEC = True
except ImportError:
    MODEL2VEC = False

from .base import Vectors


class Model2Vec(Vectors):
    """
    Builds vectors using Model2Vec.
    """

    @staticmethod
    def ismodel(path):
        """
        Checks if path is a Model2Vec model.

        Args:
            path: input path

        Returns:
            True if this is a Model2Vec model, False otherwise
        """

        try:
            # Download file and parse JSON
            path = cached_file(path_or_repo_id=path, filename="config.json")
            if path:
                with open(path, encoding="utf-8") as f:
                    config = json.load(f)
                    return config.get("model_type") == "model2vec"

        # Ignore this error - invalid repo or directory
        except (HFValidationError, OSError):
            pass

        return False

    def __init__(self, config, scoring, models):
        # Check before parent constructor since it calls loadmodel
        if not MODEL2VEC:
            raise ImportError('Model2Vec is not available - install "vectors" extra to enable')

        super().__init__(config, scoring, models)

    def loadmodel(self, path):
        return StaticModel.from_pretrained(path)

    def encode(self, data):
        # Additional model arguments
        modelargs = self.config.get("vectors", {})

        # Encode data
        return self.model.encode(data, batch_size=self.encodebatch, **modelargs)
