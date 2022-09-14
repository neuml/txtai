"""
NetworkX module
"""

import pickle

# Conditional import
try:
    import networkx as nx

    from community import community_louvain
    from networkx.algorithms.community import asyn_lpa_communities, greedy_modularity_communities

    NETWORKX = True
except ImportError:
    NETWORKX = False

from .. import __pickle__

from .base import Graph


# pylint: disable=R0904
class NetworkX(Graph):
    """
    Graph instance backed by NetworkX.
    """

    def __init__(self, config):
        super().__init__(config)

        if not NETWORKX:
            raise ImportError('NetworkX is not available - install "graph" extra to enable')

    def create(self):
        return nx.Graph()

    def count(self):
        return self.backend.number_of_nodes()

    def scan(self, attribute=None):
        # Nodes containing an attribute
        if attribute:
            return nx.subgraph_view(self.backend, filter_node=lambda x: attribute in self.node(x))

        # Return all nodes
        return self.backend

    def node(self, uid):
        return self.backend.nodes.get(uid)

    def hasnode(self, uid):
        return self.backend.has_node(uid)

    def addnode(self, uid, **attrs):
        self.backend.add_node(uid, **attrs)

    def removenode(self, uid):
        if self.hasnode(uid):
            self.backend.remove_node(uid)

    def attribute(self, uid, field):
        return self.node(uid).get(field) if self.hasnode(uid) else None

    def addattribute(self, uid, field, value):
        if self.hasnode(uid):
            self.node(uid)[field] = value

    def removeattribute(self, uid, field):
        return self.node(uid).pop(field, None) if self.hasnode(uid) else None

    def edgecount(self):
        return self.backend.number_of_edges()

    def edges(self, uid):
        edges = self.backend.adj.get(uid)
        if edges:
            return dict(sorted(edges.items(), key=lambda x: x[1]["weight"]), reverse="True").keys()

        return None

    def hasedge(self, source, target=None):
        if not target:
            edges = self.backend.adj.get(source)
            return len(edges) > 0 if edges else False

        return self.backend.has_edge(source, target)

    def addedge(self, source, target, **attrs):
        self.backend.add_edge(source, target, **attrs)

    def centrality(self):
        rank = nx.degree_centrality(self.backend)
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True))

    def pagerank(self):
        rank = nx.pagerank(self.backend, weight="weight")
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True))

    def showpath(self, source, target):
        # pylint: disable=E1121
        return nx.shortest_path(self.backend, source, target, self.distance)

    def communities(self, config):
        # Get community detection algorithm
        algorithm = config.get("algorithm")

        if algorithm == "greedy":
            communities = greedy_modularity_communities(self.backend, weight="weight", resolution=config.get("resolution", 100))
        elif algorithm == "lpa":
            communities = asyn_lpa_communities(self.backend, weight="weight", seed=0)
        else:
            communities = self.louvain(config)

        return communities

    def loadgraph(self, path):
        # Load graph network
        with open(path, "rb") as handle:
            self.backend = pickle.load(handle)

    def savegraph(self, path):
        # Save graph
        with open(path, "wb") as handle:
            pickle.dump(self.backend, handle, protocol=__pickle__)

    def louvain(self, config):
        """
        Runs the Louvain community detection algorithm.

        Args:
            config: topic configuration

        Returns:
            dictionary of {topic name:[ids]}
        """

        # Partition level to use
        level = config.get("level", "best")

        # Run community detection
        results = community_louvain.generate_dendrogram(self.backend, weight="weight", resolution=config.get("resolution", 100), random_state=0)

        # Get partition level (first or best)
        results = results[0] if level == "first" else community_louvain.partition_at_level(results, len(results) - 1)

        # Build mapping of community to list of ids
        communities = {}
        for k, v in results.items():
            communities[v] = [k] if v not in communities else communities[v] + [k]

        return communities.values()

    # pylint: disable=W0613
    def distance(self, source, target, attrs):
        """
        Computes distance between source and target nodes using weight.

        Args:
            source: source node
            target: target node
            attrs: edge attributes

        Returns:
            distance between source and target
        """

        # Distance is 1 - score. Skip minimal distances as they are near duplicates.
        distance = max(1.0 - attrs["weight"], 0.0)
        return distance if distance >= 0.15 else 1.00
