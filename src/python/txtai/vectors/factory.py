"""
Factory module
"""

from .external import ExternalVectors
from .transformers import TransformersVectors
from .words import WordVectors, WORDS


class VectorsFactory:
    """
    Methods to create Vectors models.
    """

    @staticmethod
    def create(config, scoring=None, models=None):
        """
        Create a Vectors model instance.

        Args:
            config: vector configuration
            scoring: scoring instance
            models: models cache

        Returns:
            Vectors
        """

        # Determine vector method
        method = VectorsFactory.method(config)
        if method == "external":
            return ExternalVectors(config, scoring, models)

        if method == "words":
            if not WORDS:
                # Raise error if trying to create Word Vectors without similarity extra
                raise ImportError(
                    'Word vector models are not available - install "similarity" extra to enable. Otherwise, specify '
                    + 'method="transformers" to use transformer backed models'
                )

            return WordVectors(config, scoring, models)

        # Default to TransformersVectors when configuration available
        return TransformersVectors(config, scoring, models) if config and "path" in config else None

    @staticmethod
    def method(config):
        """
        Get or derive the vector method.

        Args:
            config: vector configuration

        Returns:
            vector method
        """

        # Determine vector type (external, transformers or words)
        method = config.get("method")
        path = config.get("path")

        # Infer method from path, if blank
        if not method:
            if path:
                method = "words" if WordVectors.isdatabase(path) else "transformers"
            elif config.get("transform"):
                method = "external"

        return method
