"""
External module
"""

import types

import numpy as np

from ..util import Resolver

from .base import Vectors


class ExternalVectors(Vectors):
    """
    Loads pre-computed vectors. Pre-computed vectors allow integrating other vector models. They can also be used to efficiently
    test different model configurations without having to recompute vectors.
    """

    def __init__(self, config, scoring):
        super().__init__(config, scoring)

        # Lookup and resolve transform function
        self.transform = self.resolve(config.get("transform"))

    def load(self, path):
        return None

    def encode(self, data):
        # Call external transform function, if available and data not already an array
        if self.transform and data and not isinstance(data[0], np.ndarray):
            data = self.transform(data)

        # Cast to float32
        return data.astype(np.float32) if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)

    def resolve(self, transform):
        """
        Resolves a transform function.

        Args:
            transform: transform function

        Returns:
            resolved transform function
        """

        if transform:
            # Resolve transform instance, if necessary
            transform = Resolver()(transform) if transform and isinstance(transform, str) else transform

            # Get function or callable instance
            transform = transform if isinstance(transform, types.FunctionType) else transform()

        return transform
