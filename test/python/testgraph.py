"""
Graph module tests
"""

import os
import itertools
import tempfile
import unittest

from unittest.mock import patch

from txtai.archive import ArchiveFactory
from txtai.embeddings import Embeddings
from txtai.graph import Graph, GraphFactory
from txtai.serialize import SerializeFactory


# pylint: disable=R0904
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

    def testDatabase(self):
        """
        Test creating a Graph backed by a relational database
        """

        # Generate graph database
        path = os.path.join(tempfile.gettempdir(), "graph.sqlite")
        graph = GraphFactory.create({"backend": "rdbms", "url": f"sqlite:///{path}", "schema": "txtai"})

        # Initialize the graph
        graph.initialize()

        for x in range(5):
            graph.addnode(x, field=x)

        for x, y in itertools.combinations(range(5), 2):
            graph.addedge(x, y)

        # Test methods
        self.assertEqual(list(graph.scan()), [str(x) for x in range(5)])
        self.assertEqual(list(graph.scan(attribute="field")), [str(x) for x in range(5)])
        self.assertEqual(list(graph.filter([0]).scan()), [0])

        # Test save/load
        graph.save(None)
        graph.load(None)
        self.assertEqual(list(graph.scan()), [str(x) for x in range(5)])

        # Test remove node
        graph.delete([0])
        self.assertFalse(graph.hasnode(0))
        self.assertFalse(graph.hasedge(0))

        # Close graph
        graph.close()

    def testDefault(self):
        """
        Test embeddings default graph setting
        """

        embeddings = Embeddings(content=True, graph=True)
        embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        self.assertEqual(embeddings.graph.count(), len(self.data))

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

    def testEdges(self):
        """
        Test edges
        """

        # Create graph
        graph = GraphFactory.create({})
        graph.initialize()
        graph.addedge(0, 1)

        # Test edge exists
        self.assertTrue(graph.hasedge(0))
        self.assertTrue(graph.hasedge(0, 1))

    def testFilter(self):
        """
        Test creating filtered subgraphs
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Validate counts
        graph = self.embeddings.search("feel good story", graph=True)
        self.assertEqual(graph.count(), 3)
        self.assertEqual(graph.edgecount(), 2)

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

    @patch.dict(os.environ, {"ALLOW_PICKLE": "True"})
    def testLegacy(self):
        """
        Test loading a legacy graph in TAR format
        """

        # Create graph
        graph = GraphFactory.create({})
        graph.initialize()
        graph.addedge(0, 1)

        categories = ["C1"]
        topics = {"T1": [0, 1]}

        serializer = SerializeFactory.create("pickle", allowpickle=True)

        # Save files to temporary directory and combine into TAR
        path = os.path.join(tempfile.gettempdir(), "graph.tar")
        with tempfile.TemporaryDirectory() as directory:
            # Save graph
            serializer.save(graph.backend, f"{directory}/graph")

            # Save categories, if necessary
            serializer.save(categories, f"{directory}/categories")

            # Save topics, if necessary
            serializer.save(topics, f"{directory}/topics")

            # Pack files
            archive = ArchiveFactory.create(directory)
            archive.save(path, "tar")

        # Load loading legacy format
        graph = GraphFactory.create({})
        graph.load(path)

        # Validate graph data is correct
        self.assertEqual(graph.count(), 2)
        self.assertEqual(graph.edgecount(), 1)
        self.assertEqual(graph.topics, topics)
        self.assertEqual(graph.categories, categories)

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
        self.assertRaises(NotImplementedError, graph.addnodes, None)
        self.assertRaises(NotImplementedError, graph.removenode, None)
        self.assertRaises(NotImplementedError, graph.hasnode, None)
        self.assertRaises(NotImplementedError, graph.attribute, None, None)
        self.assertRaises(NotImplementedError, graph.addattribute, None, None, None)
        self.assertRaises(NotImplementedError, graph.removeattribute, None, None)
        self.assertRaises(NotImplementedError, graph.edgecount)
        self.assertRaises(NotImplementedError, graph.edges, None)
        self.assertRaises(NotImplementedError, graph.addedge, None, None)
        self.assertRaises(NotImplementedError, graph.addedges, None)
        self.assertRaises(NotImplementedError, graph.hasedge, None, None)
        self.assertRaises(NotImplementedError, graph.centrality)
        self.assertRaises(NotImplementedError, graph.pagerank)
        self.assertRaises(NotImplementedError, graph.showpath, None, None)
        self.assertRaises(NotImplementedError, graph.isquery, None)
        self.assertRaises(NotImplementedError, graph.parse, None)
        self.assertRaises(NotImplementedError, graph.search, None)
        self.assertRaises(NotImplementedError, graph.communities, None)
        self.assertRaises(NotImplementedError, graph.load, None)
        self.assertRaises(NotImplementedError, graph.save, None)
        self.assertRaises(NotImplementedError, graph.loaddict, None)
        self.assertRaises(NotImplementedError, graph.savedict)

    def testRelationships(self):
        """
        Test manually-provided relationships
        """

        # Create relationships for id 0
        relationships = [{"id": f"ID{x}"} for x in range(1, len(self.data))]

        # Test with content enabled
        self.embeddings.index({"id": f"ID{i}", "text": x, "relationships": relationships if i == 0 else None} for i, x in enumerate(self.data))
        self.assertEqual(len(self.embeddings.graph.edges(0)), len(self.data) - 1)

        # Test with content disabled
        config = self.config.copy()
        config["content"] = False

        embeddings = Embeddings(config)
        embeddings.index({"id": f"ID{i}", "text": x, "relationships": relationships if i == 0 else None} for i, x in enumerate(self.data))
        self.assertEqual(len(embeddings.graph.edges(0)), len(self.data) - 1)
        embeddings.close()

    def testRelationshipsInvalid(self):
        """
        Test manually-provided relationships with no matching id
        """

        # Create relationships for id 0
        relationships = [{"id": "INVALID"}]

        # Index with invalid relationship
        self.embeddings.index({"text": x, "relationships": relationships if i == 0 else None} for i, x in enumerate(self.data))

        # Validate only relationship is semantically-derived
        edges = list(self.embeddings.graph.edges(0))
        self.assertTrue(len(edges) == 1 and edges[0] != "INVALID")

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

    def testSaveDict(self):
        """
        Test loading and saving to dictionaries
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Validate counts
        graph = self.embeddings.graph
        count, edgecount = graph.count(), graph.edgecount()

        # Save and reload graph as dict
        data = graph.savedict()
        graph.loaddict(data)

        # Validate counts
        self.assertEqual(graph.count(), count)
        self.assertEqual(graph.edgecount(), edgecount)

    def testSearch(self):
        """
        Test search
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Run standard search
        results = self.embeddings.search(
            """
            MATCH (A)-[]->(B)
            RETURN A, B
        """
        )
        self.assertEqual(len(results), 3)

        # Run path search
        results = self.embeddings.search(
            """
            MATCH P=()-[]->()
            RETURN P
        """
        )
        self.assertEqual(len(results), 3)

        # Run graph search
        g = self.embeddings.search(
            """
            MATCH (A)-[]->(B)
            RETURN A, ID(B)
        """,
            graph=True,
        )
        self.assertEqual(g.count(), 3)

        # Run path search
        results = self.embeddings.search(
            """
            MATCH P=()-[]->()
            RETURN P
        """,
            graph=True,
        )
        self.assertEqual(g.count(), 3)

        # Run similar search
        results = self.embeddings.search(
            """
            MATCH P=(A)-[]->()
            WHERE SIMILAR(A, "feel good story")
            RETURN A
            ORDER BY A.score DESC
            LIMIT 1
        """,
            graph=True,
        )
        self.assertEqual(list(results.scan())[0], 4)

    def testSearchBatch(self):
        """
        Test batch search
        """

        # Create an index for the list of text
        self.embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Run standard search
        results = self.embeddings.batchsearch(
            [
                """
            MATCH (A)-[]->(B)
            RETURN A, B
        """
            ]
        )
        self.assertEqual(len(results[0]), 3)

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

    def testSubindex(self):
        """
        Test subindex
        """

        # Build data array
        data = [(uid, text, None) for uid, text in enumerate(self.data)]

        embeddings = Embeddings(
            {
                "content": True,
                "functions": [{"name": "graph", "function": "indexes.index1.graph.attribute"}],
                "expressions": [
                    {"name": "category", "expression": "graph(indexid, 'category')"},
                    {"name": "topic", "expression": "graph(indexid, 'topic')"},
                    {"name": "topicrank", "expression": "graph(indexid, 'topicrank')"},
                ],
                "indexes": {
                    "index1": {
                        "path": "sentence-transformers/nli-mpnet-base-v2",
                        "graph": {
                            "limit": 5,
                            "minscore": 0.2,
                            "batchsize": 4,
                            "approximate": False,
                            "topics": {"categories": ["News"], "stopwords": ["the"]},
                        },
                    }
                },
            }
        )

        # Create an index for the list of text
        embeddings.index(data)

        # Test function
        result = embeddings.search("select id, category, topic, topicrank from txtai where id = 0", 1)[0]

        # Check columns have a value
        self.assertIsNotNone(result["category"])
        self.assertIsNotNone(result["topic"])
        self.assertIsNotNone(result["topicrank"])

        # Update data
        data[0] = (0, "Feel good story: lottery winner announced", None)
        embeddings.upsert([data[0]])

        # Test function
        result = embeddings.search("select id, category, topic, topicrank from txtai where id = 0", 1)[0]

        # Check columns have a value
        self.assertIsNotNone(result["category"])
        self.assertIsNotNone(result["topic"])
        self.assertIsNotNone(result["topicrank"])

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
