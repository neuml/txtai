"""
API factory module
"""


class Factory:
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

        parts = api.split(".")
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)

        return m

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

        return Factory.get(api)(config)
