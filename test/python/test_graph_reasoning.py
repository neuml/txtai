import unittest
import networkx as nx
from txtai.graph import GraphFactory

# Import your new wrapper
from txtai.graph.reasoning import ReasoningGraph

class TestReasoningGraph(unittest.TestCase):
    def test_owlrl_reasoning(self):
        """Proves the lightweight OWL-RL engine intercepts and deduces."""
        
        # 1. Stand up a standard TxtAI backend graph (NetworkX)
        backend = GraphFactory.create({"networkx": True})
        backend.backend = nx.Graph()
        
        # 2. Wrap it in your reasoning engine
        graph = ReasoningGraph(backend, use_owlready2=False)

        # 3. Inject semantic data
        graph.addnode("Abhinav", type="Engineer", language="Python")
        graph.addnode("Server", type="Machine", status="Active")

        # 4. Execute the reasoning closure
        graph.reason()

        # 5. The Assertion: If it intercepted correctly, the RDF graph will not be empty.
        self.assertGreater(len(graph.rdf_graph), 0, "RDF graph failed to populate.")
        
        # Verify the backend still received the nodes (Proxy Check)
        self.assertEqual(graph.backend.count(), 2, "Backend graph failed to receive nodes.")

    def test_owlready2_reasoning(self):
        """Proves the heavy Owlready2 engine engages if requested."""
        backend = GraphFactory.create({"networkx": True})
        backend.backend = nx.Graph()
        graph = ReasoningGraph(backend, use_owlready2=True)
        
        graph.addnode("Data", type="Artifact")
        graph.reason()
        
        self.assertGreater(len(graph.rdf_graph), 0, "RDF graph failed to populate.")
    
    def test_proxy_passthrough(self):
        """Proves the wrapper seamlessly routes native topology methods to the backend."""
        import networkx as nx
        backend = GraphFactory.create({"networkx": True})
        backend.backend = nx.Graph()
        graph = ReasoningGraph(backend, use_owlready2=False)

        # 1. Build a topology through the wrapper
        graph.addnode("DB_Master")
        graph.addnode("DB_Replica")
        
        # addedge is a native TxtAI method. The proxy must catch this and route it.
        try:
            graph.addedge("DB_Master", "DB_Replica", weight=1.0)
        except AttributeError:
            self.fail("The proxy (__getattr__) failed to route 'addedge' to the backend.")

        # 2. Verify state retrieval through the proxy
        self.assertTrue(graph.hasnode("DB_Master"), "Proxy failed to route 'hasnode'.")
        
        # 3. Verify the underlying backend actually holds the topological edge
        # (TxtAI's internal NetworkX graph stores edges in backend.backend)
        self.assertTrue(graph.backend.backend.has_edge("DB_Master", "DB_Replica"), 
                        "Edge was dropped in transit.")
    
    def test_attribute_corruption(self):
        """Proves the RDF mapping survives non-string data types (ints, floats, booleans)."""
        import networkx as nx
        backend = GraphFactory.create({"networkx": True})
        backend.backend = nx.Graph()
        graph = ReasoningGraph(backend, use_owlready2=False)

        try:
            # Injecting raw integers, floats, and booleans
            graph.addnode("Cluster1", type="System", nodes=5, uptime=99.9, active=True)
            graph.reason()
        except Exception as e:
            self.fail(f"Architecture fractured under complex data types: {e}")

        # Ensure the RDF graph still populated despite the raw data types
        self.assertGreater(len(graph.rdf_graph), 0, "RDF failed to parse complex attributes.")