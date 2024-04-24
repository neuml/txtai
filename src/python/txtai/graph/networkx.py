"""
NetworkX module
"""

import pickle

# Conditional import
try:
    import networkx as nx

    from community import community_louvain
    from grandcypher import GrandCypher
    from networkx.algorithms.community import asyn_lpa_communities, greedy_modularity_communities
    from networkx.readwrite import json_graph

    NETWORKX = True
except ImportError:
    NETWORKX = False

from ..version import __pickle__

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

    def node(self, node):
        return self.backend.nodes.get(node)

    def addnode(self, node, **attrs):
        self.backend.add_node(node, **attrs)

    def addnodes(self, nodes):
        self.backend.add_nodes_from(nodes)

    def removenode(self, node):
        if self.hasnode(node):
            self.backend.remove_node(node)

    def hasnode(self, node):
        return self.backend.has_node(node)

    def attribute(self, node, field):
        return self.node(node).get(field) if self.hasnode(node) else None

    def addattribute(self, node, field, value):
        if self.hasnode(node):
            self.node(node)[field] = value

    def removeattribute(self, node, field):
        return self.node(node).pop(field, None) if self.hasnode(node) else None

    def edgecount(self):
        return self.backend.number_of_edges()

    def edges(self, node):
        edges = self.backend.adj.get(node)
        if edges:
            return dict(sorted(edges.items(), key=lambda x: x[1].get("weight", 0), reverse=True))

        return None

    def addedge(self, source, target, **attrs):
        self.backend.add_edge(source, target, **attrs)

    def addedges(self, edges):
        self.backend.add_edges_from(edges)

    def hasedge(self, source, target=None):
        if not target:
            edges = self.backend.adj.get(source)
            return len(edges) > 0 if edges else False

        return self.backend.has_edge(source, target)

    def centrality(self):
        rank = nx.degree_centrality(self.backend)
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True))

    def pagerank(self):
        rank = nx.pagerank(self.backend, weight="weight")
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True))

    def showpath(self, source, target):
        # pylint: disable=E1121
        return nx.shortest_path(self.backend, source, target, self.distance)

    def search(self, query, limit=None, graph=False):
        # Run openCypher query
        results = GrandCypher(self.backend, limit if limit else 3).run(query)

        # Transform into filtered graph
        if graph:
            nodes = set()
            for column in results.values():
                for value in column:
                    if isinstance(value, list):
                        # Path group
                        nodes.update([node for node in value if node and not isinstance(node, dict)])
                    elif value is not None:
                        # Single result
                        nodes.add(value)

            return self.filter(list(nodes))

        # Transform columnar structure into rows
        keys = list(results.keys())
        rows, count = [], len(results[keys[0]])

        for x in range(count):
            rows.append({str(key): results[key][x] for key in keys})

        return rows

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

    def loaddict(self, data):
        self.backend = json_graph.node_link_graph(data)
        self.categories, self.topics = data.get("categories"), data.get("topics")

    def savedict(self):
        data = json_graph.node_link_data(self.backend)
        data["categories"] = self.categories
        data["topics"] = self.topics

        return data

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
