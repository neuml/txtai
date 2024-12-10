"""
PGVector module
"""

import os

import numpy as np

# Conditional import
try:
    from pgvector.sqlalchemy import BIT, HALFVEC, VECTOR

    from sqlalchemy import create_engine, delete, func, text, Column, Index, Integer, MetaData, StaticPool, Table
    from sqlalchemy.orm import Session
    from sqlalchemy.schema import CreateSchema

    PGVECTOR = True
except ImportError:
    PGVECTOR = False

from .base import ANN


class PGVector(ANN):
    """
    Builds an ANN index backed by a Postgres database.
    """

    def __init__(self, config):
        super().__init__(config)

        if not PGVECTOR:
            raise ImportError('PGVector is not available - install "ann" extra to enable')

        # Database connection
        self.engine, self.database, self.connection, self.table = None, None, None, None

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
        # Commit session and connection
        self.database.commit()
        self.connection.commit()

    def close(self):
        # Parent logic
        super().close()

        # Close database connection
        if self.database:
            self.database.close()
            self.engine.dispose()

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
        table = self.setting("table", "vectors")

        # Get embedding column and index settings
        column, index = self.column()

        # Create vectors table
        self.table = Table(table, MetaData(), Column("indexid", Integer, primary_key=True, autoincrement=False), Column("embedding", column))

        # Create ANN index - inner product is equal to cosine similarity on normalized vectors
        index = Index(
            f"{table}-index",
            self.table.c["embedding"],
            postgresql_using="hnsw",
            postgresql_with=self.settings(),
            postgresql_ops={"embedding": index},
        )

        # Drop and recreate table
        if recreate:
            self.table.drop(self.connection, checkfirst=True)
            index.drop(self.connection, checkfirst=True)

        # Create table and index
        self.table.create(self.connection, checkfirst=True)
        index.create(self.connection, checkfirst=True)

    def connect(self):
        """
        Establishes a database connection. Cleans up any existing database connection first.
        """

        # Close existing connection
        if self.database:
            self.close()

        # Create engine
        self.engine = create_engine(self.setting("url", os.environ.get("ANN_URL")), poolclass=StaticPool, echo=False)
        self.connection = self.engine.connect()

        # Start database session
        self.database = Session(self.connection)

        # Initialize pgvector extension
        self.sqldialect(text("CREATE EXTENSION IF NOT EXISTS vector"))

    def schema(self):
        """
        Sets the database schema, if available.
        """

        # Set default schema, if necessary
        schema = self.setting("schema")
        if schema:
            with self.engine.begin():
                self.sqldialect(CreateSchema(schema, if_not_exists=True))

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
        Executes a SQL statement based on the current SQL dialect.

        Args:
            sql: SQL to execute
            parameters: optional bind parameters
        """

        args = (sql, parameters) if self.engine.dialect.name == "postgresql" else (text("SELECT 1"),)
        self.database.execute(*args)

    def column(self):
        """
        Gets embedding column and index definitions for the current settings.

        Returns:
            embedding column definition, index definition
        """

        if self.qbits:
            # If quantization is set, always return BIT vectors
            return BIT(self.config["dimensions"] * 8), "bit_hamming_ops"

        if self.setting("precision") == "half":
            # 16-bit HALF precision vectors
            return HALFVEC(self.config["dimensions"]), "halfvec_ip_ops"

        # Default is full 32-bit FULL precision vectors
        return VECTOR(self.config["dimensions"]), "vector_ip_ops"

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
