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
    def create(config, scoring):
        """
        Create a Vectors model instance.

        Args:
            config: vector configuration
            scoring: scoring instance

        Returns:
            Vectors
        """

        # Determine vector method
        method = VectorsFactory.method(config)
        if method == "external":
            return ExternalVectors(config, scoring)

        if method == "words":
            if not WORDS:
                # Raise error if trying to create Word Vectors without similarity extra
                raise ImportError(
                    'Word vector models are not available - install "similarity" extra to enable. Otherwise, specify '
                    + 'method="transformers" to use transformer backed models'
                )

            return WordVectors(config, scoring)

        return TransformersVectors(config, scoring)

    @staticmethod
    def method(config):
        """
        Get or derive the vector method.

        Args:
            config: vector configuration

        Returns:
            vector method
        """

        # Determine vector type (words or transformers)
        method = config.get("method")
        path = config.get("path")

        # Infer method from path, if blank
        if not method and path:
            method = "words" if WordVectors.isdatabase(path) else "transformers"

        return method
