"""
Model2Vec module
"""

import json

# Conditional import
try:
    from model2vec import StaticModel

    MODEL2VEC = True
except ImportError:
    MODEL2VEC = False

from ...util import Download, DownloadError
from ..base import Vectors


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
            path = Download()(path, "config.json")
            if path:
                with open(path, encoding="utf-8") as f:
                    config = json.load(f)
                    return config.get("model_type") == "model2vec"

        # Ignore invalid repo/directory and when HF Hub is not available
        except (DownloadError, ImportError):
            pass

        return False

    def __init__(self, config, scoring, models):
        # Check before parent constructor since it calls loadmodel
        if not MODEL2VEC:
            raise ImportError('Model2Vec is not available - install "vectors" extra to enable')

        super().__init__(config, scoring, models)

    def loadmodel(self, path):
        return StaticModel.from_pretrained(path)

    def encode(self, data, category=None):
        # Additional model arguments
        modelargs = self.config.get("vectors", {})

        # Encode data
        return self.model.encode(data, batch_size=self.encodebatch, **modelargs)
