"""
PGVector module
"""

import os

# Conditional import
try:
    from pgvector.sqlalchemy import BIT, HALFVEC, VECTOR

    from sqlalchemy import NullPool, create_engine, delete, func, text, Column, Index, Integer, MetaData, Table
    from sqlalchemy.orm import Session, scoped_session, sessionmaker
    from sqlalchemy.schema import CreateSchema

    PGVECTOR = True
except ImportError:
    PGVECTOR = False

from ..base import ANN

# Core library imports
from ...util import Library

np = Library().numpy()


# pylint: disable=R0904
class PGVector(ANN):
    """
    Builds an ANN index backed by a Postgres database.
    """

    def __init__(self, config):
        super().__init__(config)

        if not PGVECTOR:
            raise ImportError('PGVector is not available - install "ann" extra to enable')

        # Database connection
        self.engine, self.database, self.table = None, None, None

        # Scalar quantization
        quantize = self.config.get("quantize")
        self.qbits = quantize if quantize and isinstance(quantize, int) and not isinstance(quantize, bool) else None

    def load(self, path):
        # Initialize tables
        self.initialize()

    def index(self, embeddings):
        # Initialize tables
        self.initialize(recreate=True)

        # Prepare embeddings and insert rows
        self.database.execute(self.table.insert(), [{"indexid": x, "embedding": self.prepare(row)} for x, row in enumerate(embeddings)])

        # Create index
        self.createindex()

        # Add id offset and index build metadata
        self.config["offset"] = embeddings.shape[0]
        self.metadata(self.settings())

    def append(self, embeddings):
        # Prepare embeddings and insert rows
        self.database.execute(
            self.table.insert(), [{"indexid": x + self.config["offset"], "embedding": self.prepare(row)} for x, row in enumerate(embeddings)]
        )

        # Update id offset and index metadata
        self.config["offset"] += embeddings.shape[0]
        self.metadata()

    def delete(self, ids):
        self.database.execute(delete(self.table).where(self.table.c["indexid"].in_(ids)))

    def search(self, queries, limit):
        results = []
        for query in queries:
            # Run query
            query = self.database.query(self.table.c["indexid"], self.query(query)).order_by("score").limit(limit)

            # Calculate and collect scores
            results.append([(indexid, self.score(score)) for indexid, score in query])

        return results

    def count(self):
        # pylint: disable=E1102
        return self.database.query(func.count(self.table.c["indexid"])).scalar()

    def save(self, path):
        # Commit the current thread's session
        self.database.commit()

    def close(self):
        # Parent logic
        super().close()

        # Close all thread-local Sessions and dispose the engine
        if self.engine:
            self.database.remove()
            self.database = None
            self.engine.dispose()
            self.engine = None

    def initialize(self, recreate=False):
        """
        Initializes a new database session.

        Args:
            recreate: Recreates the database tables if True
        """

        # Connect to database
        self.connect()

        # Set the database schema
        self.schema()

        # Table name
        table = self.setting("table", self.defaulttable())

        # Create vectors table object
        self.table = Table(table, MetaData(), Column("indexid", Integer, primary_key=True, autoincrement=False), Column("embedding", self.column()))

        # Drop table via engine so DDL auto-commits
        if recreate:
            self.table.drop(self.engine, checkfirst=True)

        # Create table via engine so DDL auto-commits and is visible to all scoped sessions
        self.table.create(self.engine, checkfirst=True)

    def createindex(self):
        """
        Creates a index with the current settings.
        """

        # Table name
        table = self.setting("table", self.defaulttable())

        # Create ANN index - inner product is equal to cosine similarity on normalized vectors
        index = Index(
            f"{table}-index",
            self.table.c["embedding"],
            postgresql_using="hnsw",
            postgresql_with=self.settings(),
            postgresql_ops={"embedding": self.operation()},
        )

        # Create or recreate index via engine so DDL auto-commits
        index.drop(self.engine, checkfirst=True)
        index.create(self.engine, checkfirst=True)

    def connect(self):
        """
        Establishes a database connection. Cleans up any existing database connection first.
        """

        # Close existing connection
        if self.engine:
            self.close()

        # Create engine. NullPool disables connection reuse so each thread gets its own
        # fresh DBAPI connection — a prerequisite for per-thread Session isolation.
        self.engine = create_engine(self.url(), poolclass=NullPool, echo=False)

        # Create a per-thread scoped session factory. scoped_session proxies all Session
        # calls to a thread-local Session instance, eliminating the shared-Session race that
        # caused 'prepared'-state corruption when concurrent requests hit the same Session.
        self.database = scoped_session(sessionmaker(bind=self.engine))

        # Initialize pgvector extension (committed immediately via engine.begin())
        with self.engine.begin() as conn:
            self.sqldialect_conn(conn, text("CREATE EXTENSION IF NOT EXISTS vector"))

    def schema(self):
        """
        Sets the database schema, if available.
        """

        # Set default schema, if necessary
        schema = self.setting("schema")
        if schema:
            with self.engine.begin() as conn:
                self.sqldialect_conn(conn, CreateSchema(schema, if_not_exists=True))

            self.sqldialect(text("SET search_path TO :schema,public"), {"schema": schema})

    def settings(self):
        """
        Returns settings for this index.

        Returns:
            dict
        """

        return {"m": self.setting("m", 16), "ef_construction": self.setting("efconstruction", 200)}

    def sqldialect(self, sql, parameters=None):
        """
        Executes a SQL statement on the current thread's scoped Session.

        Args:
            sql: SQL to execute
            parameters: optional bind parameters
        """

        args = (sql, parameters) if self.engine.dialect.name == "postgresql" else (text("SELECT 1"),)
        self.database.execute(*args)

    def sqldialect_conn(self, conn, sql, parameters=None):
        """
        Executes a SQL statement on an explicit Connection (used inside engine.begin() blocks).

        Args:
            conn: SQLAlchemy Connection
            sql: SQL to execute
            parameters: optional bind parameters
        """

        args = (sql, parameters) if conn.dialect.name == "postgresql" else (text("SELECT 1"),)
        conn.execute(*args)

    def defaulttable(self):
        """
        Returns the default table name.

        Returns:
            default table name
        """

        return "vectors"

    def url(self):
        """
        Reads the database url parameter.

        Returns:
            database url
        """

        return self.setting("url", os.environ.get("ANN_URL"))

    def column(self):
        """
        Gets embedding column for the current settings.

        Returns:
            embedding column definition
        """

        if self.qbits:
            # If quantization is set, always return BIT vectors
            return BIT(self.config["dimensions"] * 8)

        if self.setting("precision") == "half":
            # 16-bit HALF precision vectors
            return HALFVEC(self.config["dimensions"])

        # Default is full 32-bit FULL precision vectors
        return VECTOR(self.config["dimensions"])

    def operation(self):
        """
        Gets the index operation for the current settings.

        Returns:
            index operation
        """

        if self.qbits:
            # If quantization is set, always return BIT vectors
            return "bit_hamming_ops"

        if self.setting("precision") == "half":
            # 16-bit HALF precision vectors
            return "halfvec_ip_ops"

        # Default is full 32-bit FULL precision vectors
        return "vector_ip_ops"

    def prepare(self, data):
        """
        Prepares data for the embeddings column. This method returns a bit string for bit vectors and
        the input data unmodified for float vectors.

        Args:
            data: input data

        Returns:
            data ready for the embeddings column
        """

        # Transform to a bit string when vector quantization is enabled
        if self.qbits:
            return "".join(np.where(np.unpackbits(data), "1", "0"))

        # Return original data
        return data

    def query(self, query):
        """
        Creates a query statement from an input query. This method uses hamming distance for bit vectors and
        the max_inner_product for float vectors.

        Args:
            query: input query

        Returns:
            query statement
        """

        # Prepare query embeddings
        query = self.prepare(query)

        # Bit vector query
        if self.qbits:
            return self.table.c["embedding"].hamming_distance(query).label("score")

        # Float vector query
        return self.table.c["embedding"].max_inner_product(query).label("score")

    def score(self, score):
        """
        Calculates the index score from the input score. This method returns the hamming score
        (1.0 - (hamming distance / total number of bits)) for bit vectors and the -score for
        float vectors.

        Args:
            score: input score

        Returns:
            index score
        """

        # Calculate hamming score as 1.0 - (hamming distance / total number of bits)
        # Bound score from 0 to 1
        if self.qbits:
            return min(max(0.0, 1.0 - (score / (self.config["dimensions"] * 8))), 1.0)

        # pgvector returns negative inner product since Postgres only supports ASC order index scans on operators
        return -score
