"""
Factory module
"""

from .networkx import NetworkX


class GraphFactory:
    """
    Methods to create graphs.
    """

    @staticmethod
    def create(config):
        """
        Create a graph.

        Args:
            config: graph configuration

        Returns:
            Graph
        """

        # Graph instance
        graph = None
        backend = config.get("backend", "networkx")

        # Create graph instance
        graph = NetworkX(config)

        # Store config back
        config["backend"] = backend

        return graph
