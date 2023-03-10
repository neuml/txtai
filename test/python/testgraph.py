"""
Graph module tests
"""

import os
import itertools
import tempfile
import unittest

from txtai.embeddings import Embeddings
from txtai.graph import Graph, GraphFactory


class TestGraph(unittest.TestCase):
    """
    Graph tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize test data.
        """

        cls.data = [
            "US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day",
        ]

        cls.config = {
            "path": "sentence-transformers/nli-mpnet-base-v2",
            "content": True,
            "functions": [{"name": "graph", "function": "graph.attribute"}],
            "expressions": [
                {"name": "category", "expression": "graph(indexid, 'category')"},
                {"name": "topic", "expression": "graph(indexid, 'topic')"},
                {"name": "topicrank", "expression": "graph(indexid, 'topicrank')"},
            ],
            "graph": {"limit": 5, "minscore": 0.2, "batchsize": 4, "approximate": False, "topics": {"categories": ["News"], "stopwords": ["the"]}},
        }

        # Create embeddings instance
        cls.embeddings = Embeddings(cls.config)

    def testAnalysis(self):
        """
        Test analysis methods
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Graph centrality
        graph = self.embeddings.graph
        centrality = graph.centrality()
        self.assertEqual(list(centrality.keys())[0], 5)

        # Page Rank
        pagerank = graph.pagerank()
        self.assertEqual(list(pagerank.keys())[0], 5)

        # Path between nodes
        path = graph.showpath(4, 5)
        self.assertEqual(len(path), 2)

    def testCommunity(self):
        """
        Test community detection
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Get graph reference
        graph = self.embeddings.graph

        # Rebuild topics with updated graph settings
        graph.config = {"topics": {"algorithm": "greedy"}}
        graph.addtopics()
        self.assertEqual(sum((len(graph.topics[x]) for x in graph.topics)), 6)

        graph.config = {"topics": {"algorithm": "lpa"}}
        graph.addtopics()
        self.assertEqual(sum((len(graph.topics[x]) for x in graph.topics)), 4)

    def testCustomBackend(self):
        """
        Test resolving a custom backend
        """

        graph = GraphFactory.create({"backend": "txtai.graph.NetworkX"})
        graph.initialize()
        self.assertIsNotNone(graph)

    def testCustomBackendNotFound(self):
        """
        Test resolving an unresolvable backend
        """

        with self.assertRaises(ImportError):
            graph = GraphFactory.create({"backend": "notfound.graph"})
            graph.initialize()

    def testDelete(self):
        """
        Test delete
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Delete row
        self.embeddings.delete([4])

        # Validate counts
        graph = self.embeddings.graph
        self.assertEqual(graph.count(), 5)
        self.assertEqual(graph.edgecount(), 1)
        self.assertEqual(sum((len(graph.topics[x]) for x in graph.topics)), 5)
        self.assertEqual(len(graph.categories), 6)

    def testFunction(self):
        """
        Test running graph functions with SQL
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Test function
        result = self.embeddings.search("select category, topic, topicrank from txtai where id = 0", 1)[0]

        # Check columns have a value
        self.assertIsNotNone(result["category"])
        self.assertIsNotNone(result["topic"])
        self.assertIsNotNone(result["topicrank"])

    def testFunctionReindex(self):
        """
        Test running graph functions with SQL after reindex
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Test functions reset with a reindex
        self.embeddings.reindex(self.embeddings.config)

        # Test function
        result = self.embeddings.search("select category, topic, topicrank from txtai where id = 0", 1)[0]

        # Check columns have a value
        self.assertIsNotNone(result["category"])
        self.assertIsNotNone(result["topic"])
        self.assertIsNotNone(result["topicrank"])

    def testIndex(self):
        """
        Test index
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Validate counts
        graph = self.embeddings.graph
        self.assertEqual(graph.count(), 6)
        self.assertEqual(graph.edgecount(), 2)
        self.assertEqual(len(graph.topics), 6)
        self.assertEqual(len(graph.categories), 6)

    def testNotImplemented(self):
        """
        Test exceptions for non-implemented methods
        """

        graph = Graph({})

        self.assertRaises(NotImplementedError, graph.create)
        self.assertRaises(NotImplementedError, graph.count)
        self.assertRaises(NotImplementedError, graph.scan, None)
        self.assertRaises(NotImplementedError, graph.node, None)
        self.assertRaises(NotImplementedError, graph.addnode, None)
        self.assertRaises(NotImplementedError, graph.removenode, None)
        self.assertRaises(NotImplementedError, graph.hasnode, None)
        self.assertRaises(NotImplementedError, graph.attribute, None, None)
        self.assertRaises(NotImplementedError, graph.addattribute, None, None, None)
        self.assertRaises(NotImplementedError, graph.removeattribute, None, None)
        self.assertRaises(NotImplementedError, graph.edgecount)
        self.assertRaises(NotImplementedError, graph.edges, None)
        self.assertRaises(NotImplementedError, graph.addedge, None, None)
        self.assertRaises(NotImplementedError, graph.hasedge, None, None)
        self.assertRaises(NotImplementedError, graph.centrality)
        self.assertRaises(NotImplementedError, graph.pagerank)
        self.assertRaises(NotImplementedError, graph.showpath, None, None)
        self.assertRaises(NotImplementedError, graph.communities, None)
        self.assertRaises(NotImplementedError, graph.loadgraph, None)
        self.assertRaises(NotImplementedError, graph.savegraph, None)

    def testResetTopics(self):
        """
        Test resetting of topics
        """

        # Create an index for the list of text
        self.embeddings.index([(1, "text", None)])
        self.embeddings.upsert([(1, "graph", None)])
        self.assertEqual(list(self.embeddings.graph.topics.keys()), ["graph"])

    def testSave(self):
        """
        Test save
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Generate temp file path
        index = os.path.join(tempfile.gettempdir(), "graph")

        # Save and reload index
        self.embeddings.save(index)
        self.embeddings.load(index)

        # Validate counts
        graph = self.embeddings.graph
        self.assertEqual(graph.count(), 6)
        self.assertEqual(graph.edgecount(), 2)
        self.assertEqual(sum((len(graph.topics[x]) for x in graph.topics)), 6)
        self.assertEqual(len(graph.categories), 6)

    def testSimple(self):
        """
        Test creating a simple graph
        """

        graph = GraphFactory.create({"topics": {}})

        # Initialize the graph
        graph.initialize()

        for x in range(5):
            graph.addnode(x)

        for x, y in itertools.combinations(range(5), 2):
            graph.addedge(x, y)

        # Validate counts
        self.assertEqual(graph.count(), 5)
        self.assertEqual(graph.edgecount(), 10)

        # Test missing edge
        self.assertIsNone(graph.edges(100))

        # Test topics with no text
        graph.addtopics()
        self.assertEqual(len(graph.topics), 5)

    def testUpsert(self):
        """
        Test upsert
        """

        # Update data
        self.embeddings.upsert([(0, {"text": "Canadian ice shelf collapses".split()}, None)])

        # Validate counts
        graph = self.embeddings.graph
        self.assertEqual(graph.count(), 6)
        self.assertEqual(graph.edgecount(), 2)
        self.assertEqual(sum((len(graph.topics[x]) for x in graph.topics)), 6)
        self.assertEqual(len(graph.categories), 6)
