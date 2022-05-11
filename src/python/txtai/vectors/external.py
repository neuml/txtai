"""
External module
"""

import numpy as np

from .base import Vectors


class ExternalVectors(Vectors):
    """
    Loads pre-computed vectors. Pre-computed vectors allow integrating other vector models. They can also be used to efficiently
    test different model configurations without having to recompute vectors.
    """

    def load(self, path):
        return None

    def encode(self, data):
        # Call external transform function, if available and data not already an array
        transform = self.config.get("transform")
        if transform and data and not isinstance(data[0], np.ndarray):
            data = transform(data)

        # Cast to float32
        return data.astype(np.float32) if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)
