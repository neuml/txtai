"""
RDBMS module
"""

import os

# Conditional import
try:
    from grand import Graph
    from grand.backends import SQLBackend, InMemoryCachedBackend

    from sqlalchemy import create_engine, text, StaticPool
    from sqlalchemy.schema import CreateSchema

    ORM = True
except ImportError:
    ORM = False

from .networkx import NetworkX


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

    def load(self, path):
        # Create graph instance
        self.graph, self.database = self.connect()

        # Store NetworkX compatible backend
        self.backend = self.graph.nx

    def save(self, path):
        self.database.commit()

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

        # pylint: disable=W0212
        return Graph(backend=InMemoryCachedBackend(backend, maxsize=None)), backend._connection
