"""
External module
"""

import random
import types

from ...util import Library, Resolver

from ..base import Vectors

# Core library imports
np = Library().numpy()


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

    def encode(self, data, category=None):
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
            # Save transform name
            name = transform

            # Resolve transform instance, if necessary
            transform = Resolver()(transform) if transform and isinstance(transform, str) else transform

            # Get function or callable instance
            transform = transform if isinstance(transform, types.FunctionType) else transform()

            # Validate transform function - must return a np.array or list of float arrays
            result = transform([str(random.randint(1_000, 100_000))])
            if len(result) == 0 or (not isinstance(result[0], np.ndarray) and not isinstance(result[0][0], float)):
                raise ImportError(f"Invalid transform function {name}")

        return transform
