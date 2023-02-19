"""
Factory module
"""

from ..util import Resolver

from .hub import HuggingFaceHub
from .storage import ObjectStorage, LIBCLOUD


class CloudFactory:
    """
    Methods to create Cloud instances.
    """

    @staticmethod
    def create(config):
        """
        Creates a Cloud instance.

        Args:
            config: cloud configuration

        Returns:
            Cloud
        """

        # Cloud instance
        cloud = None

        provider = config.get("provider", "")

        # Hugging Face Hub
        if provider.lower() == "huggingface-hub":
            cloud = HuggingFaceHub(config)

        # Cloud object storage
        elif ObjectStorage.isprovider(provider):
            cloud = ObjectStorage(config)

        # External provider
        elif provider:
            cloud = CloudFactory.resolve(provider, config)

        return cloud

    @staticmethod
    def resolve(backend, config):
        """
        Attempt to resolve a custom cloud backend.

        Args:
            backend: backend class
            config: configuration parameters

        Returns:
            Cloud
        """

        try:
            return Resolver()(backend)(config)

        except Exception as e:
            # Failure message
            message = f'Unable to resolve cloud backend: "{backend}".'

            # Append message if LIBCLOUD is not installed
            message += ' Cloud storage is not available - install "cloud" extra to enable' if not LIBCLOUD else ""

            raise ImportError(message) from e
