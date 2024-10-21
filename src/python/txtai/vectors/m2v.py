"""
Model2Vec module
"""

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

    def __init__(self, config, scoring, models):
        # Check before parent constructor since it calls loadmodel
        if not MODEL2VEC:
            raise ImportError('Model2Vec is not available - install "vectors" extra to enable')

        super().__init__(config, scoring, models)

    def loadmodel(self, path):
        return StaticModel.from_pretrained(path)

    def encode(self, data):
        return self.model.encode(data)
