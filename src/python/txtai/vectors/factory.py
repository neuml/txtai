"""
Factory module
"""

from .transformers import TransformersVectors
from .words import WordVectors


class VectorsFactory:
    """
    Methods to create Vectors models.
    """

    @staticmethod
    def create(config, scoring):
        """
        Create a Vectors model instance.

        Args:
            config: vector configuration
            scoring: scoring instance

        Returns:
            Vectors
        """

        # Derive vector type
        transformers = config.get("method") == "transformers"

        # Create vector model instance
        return TransformersVectors(config, scoring) if transformers else WordVectors(config, scoring)
