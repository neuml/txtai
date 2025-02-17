"""
Graph module
"""

from collections import Counter

from .topics import Topics


# pylint: disable=R0904
class Graph:
    """
    Base class for Graph instances. This class builds graph networks. Supports topic modeling
    and relationship traversal.
    """

    def __init__(self, config):
        """
        Creates a new Graph.

        Args:
            config: graph configuration
        """

        # Graph configuration
        self.config = config if config is not None else {}

        # Graph backend
        self.backend = None

        # Topic modeling
        self.categories = None
        self.topics = None

        # Transform columns
        columns = config.get("columns", {})
        self.text = columns.get("text", "text")
        self.object = columns.get("object", "object")

        # Attributes to copy - skips text/object/relationship fields - set to True to copy all
        self.copyattributes = config.get("copyattributes", False)

        # Relationships are manually-provided edges
        self.relationships = columns.get("relationships", "relationships")
        self.relations = {}

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

    def scan(self, attribute=None, data=False):
        """
        Iterates over nodes that match a criteria. If no criteria specified, all nodes
        are returned.

        Args:
            attribute: if specified, nodes having this attribute are returned
            data: if True, attribute data is also returned

        Returns:
            node id iterator if data is False or (id, attribute dictionary) iterator if data is True
        """

        raise NotImplementedError

    def node(self, node):
        """
        Get node by id. Returns None if not found.

        Args:
            node: node id

        Returns:
            graph node
        """

        raise NotImplementedError

    def addnode(self, node, **attrs):
        """
        Adds a node to the graph.

        Args:
            node: node id
            attrs: node attributes
        """

        raise NotImplementedError

    def addnodes(self, nodes):
        """
        Adds nodes to the graph.

        Args:
            nodes: list of (node, attributes) to add
        """

        raise NotImplementedError

    def removenode(self, node):
        """
        Removes a node and all it's edges from graph.

        Args:
            node: node id
        """

        raise NotImplementedError

    def hasnode(self, node):
        """
        Returns True if node found, False otherwise.

        Args:
            node: node id

        Returns:
            True if node found, False otherwise
        """

        raise NotImplementedError

    def attribute(self, node, field):
        """
        Gets a node attribute.

        Args:
            node: node id
            field: attribute name

        Returns:
            attribute value
        """

        raise NotImplementedError

    def addattribute(self, node, field, value):
        """
        Adds an attribute to node.

        Args:
            node: node id
            field: attribute name
            value: attribute value
        """

        raise NotImplementedError

    def removeattribute(self, node, field):
        """
        Removes an attribute from node.

        Args:
            node: node id
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

    def edges(self, node):
        """
        Gets edges of node by id.

        Args:
            node: node id

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

    def addedges(self, edges):
        """
        Adds an edge to graph.

        Args:
            edges: list of (source, target, attributes) to add
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

    def isquery(self, queries):
        """
        Checks if queries are supported graph queries.

        Args:
            queries: queries to check

        Returns:
            True if all the queries are supported graph queries, False otherwise
        """

        raise NotImplementedError

    def parse(self, query):
        """
        Parses a graph query into query components.

        Args:
            query: graph query

        Returns:
            query components as a dictionary
        """

        raise NotImplementedError

    def search(self, query, limit=None, graph=False):
        """
        Searches graph for nodes matching query.

        Args:
            query: graph query
            limit: maximum results
            graph: return graph results if True

        Returns:
            list of dict if graph is set to False
            filtered graph if graph is set to True
        """

        raise NotImplementedError

    def batchsearch(self, queries, limit=None, graph=False):
        """
        Searches graph for nodes matching query.

        Args:
            query: graph query
            limit: maximum results
            graph: return graph results if True

        Returns:
            list of dict if graph is set to False
            filtered graph if graph is set to True
        """

        return [self.search(query, limit, graph) for query in queries]

    def communities(self, config):
        """
        Run community detection on the graph.

        Args:
            config: configuration

        Returns:
            dictionary of {topic name:[ids]}
        """

        raise NotImplementedError

    def load(self, path):
        """
        Loads a graph at path.

        Args:
            path: path to graph
        """

        raise NotImplementedError

    def save(self, path):
        """
        Saves a graph at path.

        Args:
            path: path to save graph
        """

        raise NotImplementedError

    def loaddict(self, data):
        """
        Loads data from input dictionary into this graph.

        Args:
            data: input dictionary
        """

        raise NotImplementedError

    def savedict(self):
        """
        Saves graph data to a dictionary.

        Returns:
            dict
        """

        raise NotImplementedError

    def initialize(self):
        """
        Initialize graph instance.
        """

        if not self.backend:
            self.backend = self.create()

    def close(self):
        """
        Closes this graph.
        """

        self.backend, self.categories, self.topics = None, None, None

    def insert(self, documents, index=0):
        """
        Insert graph nodes for each document.

        Args:
            documents: list of (id, data, tags)
            index: indexid offset, used for node ids
        """

        # Initialize graph backend
        self.initialize()

        nodes = []
        for uid, document, _ in documents:
            # Manually provided relationships and attributes to copy
            relations, attributes = None, {}

            # Extract data from dictionary
            if isinstance(document, dict):
                # Extract relationships
                relations = document.get(self.relationships)

                # Attributes to copy, if any
                search = self.copyattributes if isinstance(self.copyattributes, list) else []
                attributes = {
                    k: v
                    for k, v in document.items()
                    if k not in [self.text, self.object, self.relationships] and (self.copyattributes is True or k in search)
                }

                # Require text or object field
                document = document.get(self.text, document.get(self.object))

            if document is not None:
                if isinstance(document, list):
                    # Join tokens as text
                    document = " ".join(document)

                # Create node
                nodes.append((index, {**{"id": uid, "data": document}, **attributes}))

                # Add relationships
                self.addrelations(index, relations)

                index += 1

        # Add nodes
        self.addnodes(nodes)

    def delete(self, ids):
        """
        Deletes ids from graph.

        Args:
            ids: node ids to delete
        """

        for node in ids:
            # Remove existing node, if it exists
            if self.hasnode(node):
                # Delete from topics
                topic = self.attribute(node, "topic")
                if topic and self.topics:
                    # Delete id from topic
                    self.topics[topic].remove(node)

                    # Also delete topic, if it's empty
                    if not self.topics[topic]:
                        self.topics.pop(topic)

                # Delete node
                self.removenode(node)

    def index(self, search, ids, similarity):
        """
        Build relationships between graph nodes using a score-based search function.

        Args:
            search: batch search function - takes a list of queries and returns lists of (id, scores) to use as edge weights
            ids: ids function - internal id resolver
            similarity: batch similarity function - takes a list of text and labels and returns best matches
        """

        # Add relationship edges
        self.resolverelations(ids)

        # Infer node edges using search function
        self.inferedges(self.scan(), search)

        # Label categories/topics
        if "topics" in self.config:
            self.addtopics(similarity)

    def upsert(self, search, ids, similarity=None):
        """
        Adds relationships for new graph nodes using a score-based search function.

        Args:
            search: batch search function - takes a list of queries and returns lists of (id, scores) to use as edge weights
            ids: ids function - internal id resolver
            similarity: batch similarity function - takes a list of text and labels and returns best matches
        """

        # Detect if topics processing is enabled
        hastopics = "topics" in self.config

        # Add relationship edges
        self.resolverelations(ids)

        # Infer node edges using new/updated nodes, set updated flag for topic processing, if necessary
        self.inferedges(self.scan(attribute="data"), search, {"updated": True} if hastopics else None)

        # Infer topics with topics of connected nodes
        if hastopics:
            # Infer topics if there is at least one topic, otherwise rebuild
            if self.topics:
                self.infertopics()
            else:
                self.addtopics(similarity)

    def filter(self, nodes, graph=None):
        """
        Creates a subgraph of this graph using the list of input nodes. This method creates a new graph
        selecting only matching nodes, edges, topics and categories.

        Args:
            nodes: nodes to select as a list of ids or list of (id, score) tuples
            graph: optional graph used to store filtered results

        Returns:
            graph
        """

        # Set graph if available, otherwise create a new empty graph of the same type
        graph = graph if graph else type(self)(self.config)

        # Initalize subgraph
        graph.initialize()

        nodeids = {node[0] if isinstance(node, tuple) else node for node in nodes}
        for node in nodes:
            # Unpack node and score, if available
            node, score = node if isinstance(node, tuple) else (node, None)

            # Add nodes
            graph.addnode(node, **self.node(node))

            # Add score if present
            if score is not None:
                graph.addattribute(node, "score", score)

            # Add edges
            edges = self.edges(node)
            if edges:
                for target, attributes in self.edges(node).items():
                    if target in nodeids:
                        graph.addedge(node, target, **attributes)

        # Filter categories and topics
        if self.topics:
            topics = {}
            for i, (topic, ids) in enumerate(self.topics.items()):
                ids = [x for x in ids if x in nodeids]
                if ids:
                    topics[topic] = (self.categories[i] if self.categories else None, ids)

            # Sort by number of nodes descending
            topics = sorted(topics.items(), key=lambda x: len(x[1][1]), reverse=True)

            # Copy filtered categories and topics
            graph.categories = [category for _, (category, _) in topics] if self.categories else None
            graph.topics = {topic: ids for topic, (_, ids) in topics}

        return graph

    def addrelations(self, node, relations):
        """
        Add manually-provided relationships.

        Args:
            node: node id
            relations: list of relationships to add
        """

        # Add relationships, if any
        if relations:
            if node not in self.relations:
                self.relations[node] = []

            # Add each relationship
            for relation in relations:
                # Support both dict and string ids
                relation = {"id": relation} if not isinstance(relation, dict) else relation
                self.relations[node].append(relation)

    def resolverelations(self, ids):
        """
        Resolves ids and creates edges for manually-provided relationships.

        Args:
            ids: internal id resolver
        """

        # Relationship edges
        edges = []

        # Resolve ids and create edges for relationships
        for node, relations in self.relations.items():
            # Resolve internal ids
            iids = ids(y["id"] for y in relations)

            # Add each edge
            for relation in relations:
                # Make copy of relation
                relation = relation.copy()

                # Lookup targets for relationship
                targets = iids.get(str(relation.pop("id")))

                # Create edge for each instance of id - internal id pair
                if targets:
                    for target in targets:
                        # Add weight, if not provided
                        relation["weight"] = relation.get("weight", 1.0)

                        # Add edge and all other attributes
                        edges.append((node, target, relation))

        # Add relationships
        if edges:
            self.addedges(edges)

        # Clear temporary relationship storage
        self.relations = {}

    def inferedges(self, nodes, search, attributes=None):
        """
        Infers edges for a list of nodes using a score-based search function.

        Args:
            nodes: list of nodes
            search: search function to use to identify edges
            attribute: dictionary of attributes to add to each node
        """

        # Read graph parameters
        batchsize, limit, minscore = self.config.get("batchsize", 256), self.config.get("limit", 15), self.config.get("minscore", 0.1)
        approximate = self.config.get("approximate", True)

        batch = []
        for node in nodes:
            # Get data attribute
            data = self.removeattribute(node, "data")

            # Set text field when data is a string
            if isinstance(data, str):
                self.addattribute(node, "text", data)

            # Add additional attributes, if specified
            if attributes:
                for field, value in attributes.items():
                    self.addattribute(node, field, value)

            # Skip nodes with existing edges when building an approximate network
            if not approximate or not self.hasedge(node):
                batch.append((node, data))

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

        edges = []
        for x, result in enumerate(search([data for _, data in batch], limit)):
            # Get input node id
            x, _ = batch[x]

            # Add edges for each input node id and result node id pair that meets specified criteria
            for y, score in result:
                if str(x) != str(y) and score > minscore:
                    edges.append((x, y, {"weight": score}))

        self.addedges(edges)

    def addtopics(self, similarity=None):
        """
        Identifies and adds topics using community detection.

        Args:
            similarity: similarity function for labeling categories
        """

        # Clear previous topics, if any
        self.cleartopics()

        # Use community detection to get topics
        topics = Topics(self.config["topics"])
        config = topics.config
        self.topics = topics(self)

        # Label each topic with a higher level category
        if "categories" in config and similarity:
            self.categories = []
            results = similarity(self.topics.keys(), config["categories"])
            for result in results:
                self.categories.append(config["categories"][result[0][0]])

        # Add topic-related node attributes
        for x, topic in enumerate(self.topics):
            for r, node in enumerate(self.topics[topic]):
                self.addattribute(node, "topic", topic)
                self.addattribute(node, "topicrank", r)

                if self.categories:
                    self.addattribute(node, "category", self.categories[x])

    def cleartopics(self):
        """
        Clears topic fields from all nodes.
        """

        # Clear previous topics, if any
        if self.topics:
            for node in self.scan():
                self.removeattribute(node, "topic")
                self.removeattribute(node, "topicrank")

                if self.categories:
                    self.removeattribute(node, "category")

            self.topics, self.categories = None, None

    def infertopics(self):
        """
        Infers topics for all nodes with an "updated" attribute. This method analyzes the direct node
        neighbors and set the most commonly occuring topic and category for each node.
        """

        # Iterate over nodes missing topic attribute (only occurs for new nodes)
        for node in self.scan(attribute="updated"):
            # Remove updated attribute
            self.removeattribute(node, "updated")

            # Get list of neighboring nodes
            ids = self.edges(node)
            ids = ids.keys() if ids else None

            # Infer topic
            topic = Counter(self.attribute(x, "topic") for x in ids).most_common(1)[0][0] if ids else None
            if topic:
                # Add id to topic list and set topic attribute
                self.topics[topic].append(node)
                self.addattribute(node, "topic", topic)

                # Set topic rank
                self.addattribute(node, "topicrank", len(self.topics[topic]) - 1)

                # Infer category
                category = Counter(self.attribute(x, "category") for x in ids).most_common(1)[0][0]
                self.addattribute(node, "category", category)
