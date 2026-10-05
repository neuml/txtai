"""
API factory module
"""

from ..util import Resolver

from .base import API


class APIFactory:
    """
    API factory. Creates new API instances.
    """

    @staticmethod
    def get(path, base=None):
        """
        Gets a new instance of a class.

        Args:
            path: class path
            base: optional required base class

        Returns:
            Class
        """

        return Resolver()(path, base)

    @staticmethod
    def create(config, api):
        """
        Creates a new API instance.

        Args:
            config: API configuration
            api: API instance class

        Returns:
            API instance
        """

        return APIFactory.get(api, API)(config)
