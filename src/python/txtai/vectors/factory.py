"""
Factory module
"""

from .transformers import TransformersVectors
from .words import WordVectors, WORDS


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

        if not transformers and not WORDS:
            # Raise error if trying to create Word Vectors without similarity extra
            raise ImportError(
                'Word vector models are not available - install "similarity" extra to enable. Otherwise, specify '
                + 'method="transformers" to use transformer backed models'
            )

        # Create vector model instance
        return TransformersVectors(config, scoring) if transformers else WordVectors(config, scoring)
