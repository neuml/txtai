"""
API factory module
"""

from ..util import Resolver


class APIFactory:
    """
    API factory. Creates new API instances.
    """

    @staticmethod
    def get(api):
        """
        Gets a new instance of api class.

        Args:
            api: API instance class

        Returns:
            API
        """

        return Resolver()(api)

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

        return APIFactory.get(api)(config)
