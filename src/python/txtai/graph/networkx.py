"""
NetworkX module
"""

import os

from tempfile import TemporaryDirectory

# Conditional import
try:
    import networkx as nx

    from networkx.algorithms.community import asyn_lpa_communities, greedy_modularity_communities, louvain_partitions
    from networkx.readwrite import json_graph

    NETWORKX = True
except ImportError:
    NETWORKX = False

from ..archive import ArchiveFactory
from ..serialize import SerializeError, SerializeFactory

from .base import Graph
from .query import Query


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

    def scan(self, attribute=None, data=False):
        # Full graph
        graph = self.backend

        # Filter graph to nodes having a specified attribute
        if attribute:
            graph = nx.subgraph_view(self.backend, filter_node=lambda x: attribute in self.node(x))

        # Return either list of matching ids or tuple of (id, attribute dictionary)
        return graph.nodes(data=True) if data else graph

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
        if target is None:
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

    def isquery(self, queries):
        return Query().isquery(queries)

    def parse(self, query):
        return Query().parse(query)

    def search(self, query, limit=None, graph=False):
        # Run graph query
        results = Query()(self, query, limit)

        # Transform into filtered graph
        if graph:
            nodes = set()
            for column in results.values():
                for value in column:
                    if isinstance(value, list):
                        # Path group
                        nodes.update([node for node in value if node and not isinstance(node, dict)])
                    elif isinstance(value, dict):
                        # Nodes by id attribute
                        nodes.update(uid for uid, attr in self.scan(data=True) if attr["id"] == value["id"])
                    elif value is not None:
                        # Single node id
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

    def load(self, path):
        try:
            # Load graph data
            data = SerializeFactory.create().load(path)

            # Add data to graph
            self.backend = self.create()
            self.backend.add_nodes_from(data["nodes"])
            self.backend.add_edges_from(data["edges"])

            # Load categories
            self.categories = data.get("categories")

            # Load topics
            self.topics = data.get("topics")

        except SerializeError:
            # Backwards compatible support for legacy TAR format
            self.loadtar(path)

    def save(self, path):
        # Save graph data
        SerializeFactory.create().save(
            {
                "nodes": [(uid, self.node(uid)) for uid in self.scan()],
                "edges": list(self.backend.edges(data=True)),
                "categories": self.categories,
                "topics": self.topics,
            },
            path,
        )

    def loaddict(self, data):
        self.backend = json_graph.node_link_graph(data, name="indexid")
        self.categories, self.topics = data.get("categories"), data.get("topics")

    def savedict(self):
        data = json_graph.node_link_data(self.backend, name="indexid")
        data["categories"] = self.categories
        data["topics"] = self.topics

        return data

    def louvain(self, config):
        """
        Runs the Louvain community detection algorithm.

        Args:
            config: topic configuration

        Returns:
            list of [ids] per community
        """

        # Partition level to use
        level = config.get("level", "best")

        # Run community detection
        results = list(louvain_partitions(self.backend, weight="weight", resolution=config.get("resolution", 100), seed=0))

        # Get partition level (first or best)
        return results[0] if level == "first" else results[-1]

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

    def loadtar(self, path):
        """
        Loads a graph from the legacy TAR file.

        Args:
            path: path to graph
        """

        # Pickle serialization - backwards compatible
        serializer = SerializeFactory.create("pickle")

        # Extract files to temporary directory and load content
        with TemporaryDirectory() as directory:
            # Unpack files
            archive = ArchiveFactory.create(directory)
            archive.load(path, "tar")

            # Load graph backend
            self.backend = serializer.load(f"{directory}/graph")

            # Load categories, if necessary
            path = f"{directory}/categories"
            if os.path.exists(path):
                self.categories = serializer.load(path)

            # Load topics, if necessary
            path = f"{directory}/topics"
            if os.path.exists(path):
                self.topics = serializer.load(path)
