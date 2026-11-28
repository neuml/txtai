"""
RDBMS module
"""

import os

# Conditional import
try:
    from grand import Graph
    from grand.backends import SQLBackend, InMemoryCachedBackend
    from grand.dialects import NetworkXDialect

    from sqlalchemy import create_engine, text, StaticPool
    from sqlalchemy.schema import CreateSchema

    ORM = True
except ImportError:
    ORM = False

from ..serialize import SerializeError, SerializeFactory

from .networkx import NetworkX


class DegreeView:
    """
    Read-only view of (node, degree) pairs, mirroring networkx DegreeView.
    """

    def __init__(self, degrees):
        self.degrees = degrees

    def __getitem__(self, node):
        """Returns the degree of a node."""
        return self.degrees[node]

    def __iter__(self):
        """Iterates over (node, degree) pairs."""
        return iter(self.degrees.items())

    def __len__(self):
        """Returns the number of nodes."""
        return len(self.degrees)

    def __contains__(self, node):
        """Returns True if the node is in this view."""
        return node in self.degrees


class RDBMSDialect(NetworkXDialect):
    """
    NetworkX dialect for the RDBMS backend.

    grand's NetworkXDialect.degree takes no `weight` argument and passes node
    lists into the cached backend, which rejects unhashable arguments. This
    subclass computes degrees per node from edge metadata instead, which fixes
    community detection (louvain/greedy/lpa) and centrality on this backend.
    """

    def degree(self, nbunch=None, weight=None):
        # Single node - return degree as a number
        if nbunch is not None and (isinstance(nbunch, str) or not hasattr(nbunch, "__iter__")):
            return self._degree(nbunch, weight)

        # All nodes or a subset - return (node, degree) pairs
        nodes = self._nodes(nbunch)
        return DegreeView({node: self._degree(node, weight) for node in nodes})

    def in_degree(self, nbunch=None, weight=None):
        # Single node - return degree as a number
        if nbunch is not None and (isinstance(nbunch, str) or not hasattr(nbunch, "__iter__")):
            return self._degree(nbunch, weight, "in")

        # All nodes or a subset - return (node, degree) pairs
        nodes = self._nodes(nbunch)
        return DegreeView({node: self._degree(node, weight, "in") for node in nodes})

    def out_degree(self, nbunch=None, weight=None):
        # Single node - return degree as a number
        if nbunch is not None and (isinstance(nbunch, str) or not hasattr(nbunch, "__iter__")):
            return self._degree(nbunch, weight, "out")

        # All nodes or a subset - return (node, degree) pairs
        nodes = self._nodes(nbunch)
        return DegreeView({node: self._degree(node, weight, "out") for node in nodes})

    def has_edge(self, u, v):
        # SQLBackend stores node ids as strings, so the default
        # `v in self._adj[u]` lookup misses integer ids.
        try:
            return self.parent.backend.has_edge(str(u), str(v))
        except KeyError:
            return False

    def remove_edge(self, u, v):
        # Edge removal is not supported by the RDBMS backend, matching the
        # base dialect.
        raise NotImplementedError

    def _nodes(self, nbunch):
        return list(self.parent.backend.all_nodes_as_iterable()) if nbunch is None else list(nbunch)

    def _degree(self, node, weight=None, direction=None):
        # Directed graphs sum in-degree and out-degree
        if direction is None and self.parent.backend.is_directed():
            return self._degree(node, weight, "in") + self._degree(node, weight, "out")

        if direction == "in":
            neighbors = self.parent.backend.get_node_predecessors(node, include_metadata=True)
        elif direction == "out":
            neighbors = self.parent.backend.get_node_successors(node, include_metadata=True)
        else:
            neighbors = self.parent.backend.get_node_neighbors(node, include_metadata=True)

        if weight is None:
            return len(neighbors)

        # Weighted degree sums edge weights, defaulting to 1 when unset
        return sum((metadata or {}).get(weight, 1) for metadata in neighbors.values())


class RDBMS(NetworkX):
    """
    Graph instance backed by a relational database.
    """

    def __init__(self, config):
        # Check before super() in case those required libraries are also not available
        if not ORM:
            raise ImportError('RDBMS is not available - install "graph" extra to enable')

        super().__init__(config)

        # Graph and database instances
        self.graph = None
        self.database = None

    def __del__(self):
        if hasattr(self, "database") and self.database:
            self.database.close()

    def create(self):
        # Create graph instance
        self.graph, self.database = self.connect()

        # Clear previous graph, if available
        for table in [self.config.get("nodes", "nodes"), self.config.get("edges", "edges")]:
            self.database.execute(text(f"DELETE FROM {table}"))

        # Return NetworkX compatible backend
        return self.graph.nx

    def scan(self, attribute=None, data=False):
        if attribute:
            for node in self.backend:
                attributes = self.node(node)
                if attribute in attributes:
                    yield (node, attributes) if data else node
        else:
            yield from super().scan(attribute, data)

    def addattribute(self, node, field, value):
        if self.hasnode(node):
            self.graph.backend.add_node(node, {field: value})

    def addedge(self, source, target, **attrs):
        # Store each undirected edge once in a canonical orientation so
        # (source, target) and (target, source) share one database row.
        source, target = self._canonical(source, target)
        super().addedge(source, target, **attrs)

    def addedges(self, edges):
        # Canonicalize orientation, then drop in-batch duplicates (last write
        # wins, like NetworkX) before the bulk insert.
        canonical = {}
        for edge in edges:
            source, target = edge[0], edge[1]
            attrs = edge[2] if len(edge) > 2 else {}
            source, target = self._canonical(source, target)
            canonical[(str(source), str(target))] = (source, target, attrs)

        if canonical:
            # Add edges in bulk (same as NetworkX.addedges)
            self.backend.add_edges_from(list(canonical.values()))

    def _canonical(self, source, target):
        # Undirected edges have no direction - store each pair once in a
        # canonical orientation so (u, v) and (v, u) share one database row
        # and edge counts match NetworkX.
        if not self.backend.is_directed() and str(source) > str(target):
            return target, source
        return source, target

    def load(self, path):
        # Create graph instance
        self.graph, self.database = self.connect()

        # Store NetworkX compatible backend
        self.backend = self.graph.nx

        # Load topics and categories, if previously saved
        if path:
            try:
                data = SerializeFactory.create().load(path)
                self.categories = data.get("categories")
                self.topics = data.get("topics")
            except (OSError, SerializeError, ValueError):
                # No sidecar file yet (e.g. legacy database) - topics stay unset
                pass

    def save(self, path):
        self.database.commit()

        # Persist topics and categories alongside the database, mirroring the
        # NetworkX backend. Nodes and edges already live in the database.
        if path:
            SerializeFactory.create().save({"categories": self.categories, "topics": self.topics}, path)

    def close(self):
        # Parent logic
        super().close()

        # Close database connection
        self.database.close()

    def filter(self, nodes, graph=None):
        return super().filter(nodes, graph if graph else NetworkX(self.config))

    def connect(self):
        """
        Connects to a graph backed by a relational database.

        Args:
            Graph database instance
        """

        # Keyword arguments for SQLAlchemy
        kwargs = {"poolclass": StaticPool, "echo": False}
        url = self.config.get("url", os.environ.get("GRAPH_URL"))

        # Set default schema, if necessary
        schema = self.config.get("schema")
        if schema:
            # Check that schema exists
            engine = create_engine(url)
            with engine.begin() as connection:
                connection.execute(CreateSchema(schema, if_not_exists=True) if "postgresql" in url else text("SELECT 1"))

            # Set default schema
            kwargs["connect_args"] = {"options": f'-c search_path="{schema}"'} if "postgresql" in url else {}

        backend = SQLBackend(
            db_url=url,
            node_table_name=self.config.get("nodes", "nodes"),
            edge_table_name=self.config.get("edges", "edges"),
            sqlalchemy_kwargs=kwargs,
        )

        graph = Graph(backend=InMemoryCachedBackend(backend, maxsize=None))

        # grand's NetworkXDialect.degree takes no `weight` argument and passes
        # node lists into the cached backend, which rejects unhashable arguments.
        graph.attach_dialect("nx", RDBMSDialect)

        # pylint: disable=W0212
        return graph, backend._connection
