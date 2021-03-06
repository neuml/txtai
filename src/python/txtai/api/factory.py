"""
API factory module
"""


class Factory:
    """
    API factory. Creates new API instances.
    """

    @staticmethod
    def get(atype):
        """
        Gets a new instance of atype.

        Args:
            atype: API instance class

        Returns:
            instance of atype
        """

        parts = atype.split(".")
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)

        return m
