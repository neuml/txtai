from rdflib import Graph as RDFGraph, URIRef, Literal
from rdflib.namespace import RDF, RDFS
from owlrl import DeductiveClosure, OWLRL_Semantics

# Import the base structure you located
from txtai.graph.base import Graph

# The Dependency Shield
try:
    import owlready2
    OWLREADY2_AVAILABLE = True
except ImportError:
    OWLREADY2_AVAILABLE = False

class ReasoningGraph:
    """
    Graph extension that provides semantic reasoning capabilities.
    Defaults to OWL-RL. Upgrades to Owlready2 if installed and requested.

    Proxy wrapper that adds semantic reasoning to any TxtAI graph backend.
    """
    def __init__(self, backend_graph, use_owlready2=False):
        # 1. Store the actual functional graph (e.g., NetworkX graph instance)
        self.backend = backend_graph
        
        # 2. Initialize the semantic shadow graph
        self.rdf_graph = RDFGraph()
        
        # 3. Arm the heavy engine
        self.use_owlready2 = use_owlready2 and OWLREADY2_AVAILABLE
        if self.use_owlready2:
            self.onto = owlready2.get_ontology("http://txtai.reasoning/onto.owl")

    def addnode(self, node_id, **attrs):
        """
        Intercepts node addition to mirror state into semantic engines.
        """
        # Execute the native TxtAI addition on the wrapped backend
        self.backend.addnode(node_id, **attrs)

        # Mirror into RDF
        node_uri = URIRef(str(node_id))
        if "type" in attrs:
            self.rdf_graph.add((node_uri, RDF.type, URIRef(attrs["type"])))
        
        for key, value in attrs.items():
            if key != "type":
                self.rdf_graph.add((node_uri, URIRef(key), Literal(value)))
        
        # Mirror into Owlready2 ontology
        if self.use_owlready2:
            with self.onto:
                node_type = attrs.get("type", "Thing")
                if not getattr(self.onto, node_type, None):
                    owlready2.types.new_class(node_type, (owlready2.Thing,))
                
                individual = getattr(self.onto, node_type)(str(node_id))
                for key, value in attrs.items():
                    if key != "type":
                        setattr(individual, key, value)

    def addedge(self, source, target, **attrs):
        """
        Explicitly routes edge creation to bypass the base class NotImplementedError.
        """
        return self.backend.addedge(source, target, **attrs)

    def __getattr__(self, name):
        """
        Proxy all other unrecognized method calls to the backend graph.
        """
        return getattr(self.backend, name)
    
    def reason(self):
        """
        Executes logical deduction based on the active engine.
        """
        if self.use_owlready2:
            with self.onto:
                owlready2.sync_reasoner()
        else:
            DeductiveClosure(OWLRL_Semantics).expand(self.rdf_graph)