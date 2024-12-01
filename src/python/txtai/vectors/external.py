"""
External module
"""

import types

import numpy as np

from ..util import Resolver

from .base import Vectors


class External(Vectors):
    """
    Builds vectors using an external method. This can be a local function or an external API call.
    """

    def __init__(self, config, scoring, models):
        super().__init__(config, scoring, models)

        # Lookup and resolve transform function
        self.transform = self.resolve(config.get("transform"))

    def loadmodel(self, path):
        return None

    def encode(self, data):
        # Call external transform function, if available and data not already an array
        # Batching is handed by the external transform function
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
