"""
PickleEncoder module
"""

import pickle

from .base import Encoder


class PickleEncoder(Encoder):
    """
    Encodes and decodes objects using the Python pickle package.
    """

    def encode(self, obj):
        # Pickle object
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        # Unpickle to object
        return pickle.loads(data) if data else None
