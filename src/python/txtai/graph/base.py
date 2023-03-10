"""
Graph module
"""

import os
import pickle

from collections import Counter
from tempfile import TemporaryDirectory

from .. import __pickle__

from ..archive import ArchiveFactory

from .topics import Topics


# pylint: disable=R0904
class Graph:
    """
    Base class for Graph instances.
    """

    def __init__(self, config):
        """
        Creates a new Graph.

        Args:
            config: graph configuration
        """

        # Graph configuration
        self.config = config

        # Graph backend
        self.backend = None

        # Topic modeling
        self.categories = None
        self.topics = None

    def create(self):
        """
        Creates the graph network.
        """

        raise NotImplementedError

    def count(self):
        """
        Returns the total number of nodes in graph.

        Returns:
            total nodes in graph
        """

        raise NotImplementedError

    def scan(self, attribute=None):
        """
        Iterates over nodes that match a criteria. If no criteria specified, all nodes
        are returned.

        Args:
            attribute: if specified, nodes having this attribute are returned

        Returns:
            node iterator
        """

        raise NotImplementedError

    def node(self, uid):
        """
        Get node with id uid. Returns None if not found.

        Args:
            uid: node id

        Returns:
            graph node
        """

        raise NotImplementedError

    def addnode(self, uid, **attrs):
        """
        Adds a node to the graph.

        Args:
            uid: node id
            attrs: node attributes
        """

        raise NotImplementedError

    def removenode(self, uid):
        """
        Removes a node and all it's edges from graph.

        Args:
            uid: node id
        """

        raise NotImplementedError

    def hasnode(self, uid):
        """
        Returns True if node found, False otherwise.

        Args:
            uid: node id

        Returns:
            True if node found, False otherwise
        """

        raise NotImplementedError

    def attribute(self, uid, field):
        """
        Gets a node attribute.

        Args:
            uid: node id
            field: attribute name

        Returns:
            attribute value
        """

        raise NotImplementedError

    def addattribute(self, uid, field, value):
        """
        Adds an attribute to node.

        Args:
            uid: node id
            field: attribute name
            value: attribute value
        """

        raise NotImplementedError

    def removeattribute(self, uid, field):
        """
        Removes an attribute from node.

        Args:
            uid: node id
            field: attribute name

        Returns:
            attribute value or None if not present
        """

        raise NotImplementedError

    def edgecount(self):
        """
        Returns the total number of edges.

        Returns:
            total number of edges in graph
        """

        raise NotImplementedError

    def edges(self, uid):
        """
        Gets edges of node by id.

        Args:
            uid: node id

        Returns:
            list of edge node ids
        """

        raise NotImplementedError

    def addedge(self, source, target, **attrs):
        """
        Adds an edge to graph.

        Args:
            source: node 1 id
            target: node 2 id
        """

        raise NotImplementedError

    def hasedge(self, source, target=None):
        """
        Returns True if edge found, False otherwise. If target is None, this method
        returns True if any edge is found.

        Args:
            source: node 1 id
            target: node 2 id

        Returns:
            True if edge found, False otherwise
        """

        raise NotImplementedError

    def centrality(self):
        """
        Runs a centrality algorithm on the graph.

        Returns:
            dict of {node id: centrality score}
        """

        raise NotImplementedError

    def pagerank(self):
        """
        Runs the pagerank algorithm on the graph.

        Returns:
            dict of {node id, page rank score}
        """

        raise NotImplementedError

    def showpath(self, source, target):
        """
        Gets the shortest path between source and target.

        Args:
            source: start node id
            target: end node id

        Returns:
            list of node ids representing the shortest path
        """

        raise NotImplementedError

    def communities(self, config):
        """
        Run community detection on the graph.

        Args:
            config: configuration

        Returns:
            dictionary of {topic name:[ids]}
        """

        raise NotImplementedError

    def loadgraph(self, path):
        """
        Loads a graph backend at path.

        Args:
            path: path to graph backend
        """

        raise NotImplementedError

    def savegraph(self, path):
        """
        Saves graph backend to path.

        Args:
            path: path to save graph backend
        """

        raise NotImplementedError

    def initialize(self):
        """
        Initialize graph instance.
        """

        if not self.backend:
            self.backend = self.create()

    def load(self, path):
        """
        Loads a graph at path.

        Args:
            path: path to graph
        """

        # Extract files to temporary directory and load content
        with TemporaryDirectory() as directory:
            # Unpack files
            archive = ArchiveFactory.create(directory)
            archive.load(path, "tar")

            # Load graph backend
            self.loadgraph(f"{directory}/graph")

            # Load categories, if necessary
            path = f"{directory}/categories"
            if os.path.exists(path):
                with open(path, "rb") as handle:
                    self.categories = pickle.load(handle)

            # Load topics, if necessary
            path = f"{directory}/topics"
            if os.path.exists(path):
                with open(path, "rb") as handle:
                    self.topics = pickle.load(handle)

    def save(self, path):
        """
        Saves a graph at path.

        Args:
            path: path to save graph
        """

        # Save files to temporary directory and combine into TAR
        with TemporaryDirectory() as directory:
            # Save graph
            self.savegraph(f"{directory}/graph")

            # Save categories, if necessary
            if self.categories:
                with open(f"{directory}/categories", "wb") as handle:
                    pickle.dump(self.categories, handle, protocol=__pickle__)

            # Save topics, if necessary
            if self.topics:
                with open(f"{directory}/topics", "wb") as handle:
                    pickle.dump(self.topics, handle, protocol=__pickle__)

            # Pack files
            archive = ArchiveFactory.create(directory)
            archive.save(path, "tar")

    def insert(self, documents, index=0):
        """
        Insert graph nodes for each document.

        Args:
            documents: list of (id, data, tags)
            index: indexid offset, used for node ids
        """

        # Initialize graph backend
        self.initialize()

        for _, document, _ in documents:
            if isinstance(document, dict):
                # Require text or object field
                document = document.get("text", document.get("object"))

            if document is not None:
                if isinstance(document, list):
                    # Join tokens as text
                    document = " ".join(document)

                # Create node
                self.addnode(index, data=document)
                index += 1

    def delete(self, ids):
        """
        Deletes ids from graph.

        Args:
            ids: node ids to delete
        """

        for uid in ids:
            # Remove existing node, if it exists
            if self.hasnode(uid):
                # Delete from topics
                topic = self.attribute(uid, "topic")
                if topic and self.topics:
                    # Delete id from topic
                    self.topics[topic].remove(uid)

                    # Also delete topic, if it's empty
                    if not self.topics[topic]:
                        self.topics.pop(topic)

                # Delete node
                self.removenode(uid)

    def index(self, search, similarity):
        """
        Build relationships between graph nodes using a score-based search function.

        Args:
            search: batch search function - takes a list of queries and returns lists of (id, scores) to use as edge weights
            similarity: batch similarity function - takes a list of text and labels and returns best matches
        """

        # Add node edges
        self.addedges(self.scan(), search)

        # Label categories/topics
        if "topics" in self.config:
            self.addtopics(similarity)

    def upsert(self, search, similarity=None):
        """
        Adds relationships for new graph nodes using a score-based search function.

        Args:
            search: batch search function - takes a list of queries and returns lists of (id, scores) to use as edge weights
            similarity: batch similarity function - takes a list of text and labels and returns best matches
        """

        # Detect if topics processing is enabled
        hastopics = "topics" in self.config

        # Add node edges using new/updated nodes, set updated flag for topic processing, if necessary
        self.addedges(self.scan(attribute="data"), search, {"updated": True} if hastopics else None)

        # Infer topics with topics of connected nodes
        if hastopics:
            # Infer topics if there is at least one topic, otherwise rebuild
            if self.topics:
                self.infertopics()
            else:
                self.addtopics(similarity)

    def addedges(self, nodes, search, attributes=None):
        """
        Adds edges for a list of nodes using a score-based search function.

        Args:
            nodes: list of nodes
            search: search function to use to identify edges
            attribute: dictionary of attributes to add to each node
        """

        # Read graph parameters
        batchsize, limit, minscore = self.config.get("batchsize", 256), self.config.get("limit", 15), self.config.get("minscore", 0.1)
        approximate = self.config.get("approximate", True)

        batch = []
        for uid in nodes:
            # Get data attribute
            data = self.removeattribute(uid, "data")

            # Set text field when data is a string
            if isinstance(data, str):
                self.addattribute(uid, "text", data)

            # Add additional attributes, if specified
            if attributes:
                for field, value in attributes.items():
                    self.addattribute(uid, field, value)

            # Skip nodes with existing edges when building an approximate network
            if not self.hasedge(uid) or not approximate:
                batch.append((uid, data))

            # Process batch
            if len(batch) == batchsize:
                self.addbatch(search, batch, limit, minscore)
                batch = []

        if batch:
            self.addbatch(search, batch, limit, minscore)

    def addbatch(self, search, batch, limit, minscore):
        """
        Adds batch of documents to graph. This method runs the search function for each item in batch
        and adds node edges between the input and each search result.

        Args:
            search: search function to use to identify edges
            batch: batch to add
            limit: max edges to add per node
            minscore: min score to add node edge
        """

        for x, result in enumerate(search([data for _, data in batch], limit)):
            # Get input node id
            x, _ = batch[x]

            # Add edges for each input uid and result uid pair that meets specified criteria
            for y, score in result:
                if x != y and score > minscore and not self.hasedge(x, y):
                    self.addedge(x, y, weight=score)

    def addtopics(self, similarity=None):
        """
        Identifies and adds topics using community detection.

        Args:
            similarity: similarity function for labeling categories
        """

        # Clear previous topics, if any
        self.cleartopics()

        # Use community detection to get topics
        config = self.config["topics"]
        topics = Topics(config)
        self.topics = topics(self)

        # Label each topic with a higher level category
        if "categories" in config and similarity:
            self.categories = []
            results = similarity(self.topics.keys(), config["categories"])
            for result in results:
                self.categories.append(config["categories"][result[0][0]])

        # Add topic-related node attributes
        for x, topic in enumerate(self.topics):
            for r, uid in enumerate(self.topics[topic]):
                self.addattribute(uid, "topic", topic)
                self.addattribute(uid, "topicrank", r)

                if self.categories:
                    self.addattribute(uid, "category", self.categories[x])

    def cleartopics(self):
        """
        Clears topic fields from all nodes.
        """

        # Clear previous topics, if any
        if self.topics:
            for uid in self.scan():
                self.removeattribute(uid, "topic")
                self.removeattribute(uid, "topicrank")

                if self.categories:
                    self.removeattribute(uid, "category")

            self.topics, self.categories = None, None

    def infertopics(self):
        """
        Infers topics for all nodes with an "updated" attribute. This method analyzes the direct node
        neighbors and set the most commonly occuring topic and category for each node.
        """

        # Iterate over nodes missing topic attribute (only occurs for new nodes)
        for uid in self.scan(attribute="updated"):
            # Remove updated attribute
            self.removeattribute(uid, "updated")

            # Get list of neighboring nodes
            ids = self.edges(uid)

            # Infer topic
            topic = Counter(self.attribute(x, "topic") for x in ids).most_common(1)[0][0]
            if topic:
                # Add id to topic list and set topic attribute
                self.topics[topic].append(uid)
                self.addattribute(uid, "topic", topic)

                # Set topic rank
                self.addattribute(uid, "topicrank", len(self.topics[topic]))

                # Infer category
                category = Counter(self.attribute(x, "category") for x in ids).most_common(1)[0][0]
                self.addattribute(uid, "category", category)
