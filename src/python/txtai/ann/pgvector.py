"""
PGVector module
"""

import os

# Conditional import
try:
    from pgvector.sqlalchemy import Vector

    from sqlalchemy import create_engine, delete, text, Column, Index, Integer, MetaData, StaticPool, Table
    from sqlalchemy.orm import Session

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

        # Create engine
        self.engine = create_engine(self.setting("url", os.environ.get("ANN_URL")), poolclass=StaticPool, echo=False)

        # Initialize pgvector extension
        self.database = Session(self.engine)
        self.database.execute(text("CREATE EXTENSION IF NOT EXISTS vector" if self.engine.dialect.name == "postgresql" else "SELECT 1"))
        self.database.commit()

        # Table instance
        self.table = None

    def load(self, path):
        # Reset database to original checkpoint
        self.database.rollback()

        # Initialize tables
        self.initialize()

    def index(self, embeddings):
        # Initialize tables
        self.initialize(recreate=True)

        self.database.execute(self.table.insert(), [{"indexid": x, "embedding": row} for x, row in enumerate(embeddings)])

        # Add id offset and index build metadata
        self.config["offset"] = embeddings.shape[0]
        self.metadata(self.settings())

    def append(self, embeddings):
        self.database.execute(self.table.insert(), [{"indexid": x + self.config["offset"], "embedding": row} for x, row in enumerate(embeddings)])

        # Update id offset and index metadata
        self.config["offset"] += embeddings.shape[0]
        self.metadata()

    def delete(self, ids):
        self.database.execute(delete(self.table).where(self.table.c["indexid"].in_(ids)))

    def search(self, queries, limit):
        results = []
        for query in queries:
            # Run query
            query = (
                self.database.query(self.table.c["indexid"], self.table.c["embedding"].max_inner_product(query).label("score"))
                .order_by("score")
                .limit(limit)
            )

            # pgvector returns negative inner product since Postgres only supports ASC order index scans on operators
            results.append([(indexid, -score) for indexid, score in query])

        return results

    def count(self):
        return self.database.query(self.table.c["indexid"]).count()

    def save(self, path):
        self.database.commit()

    def close(self):
        # Parent logic
        super().close()

        # Close database connection
        self.database.close()

    def initialize(self, recreate=False):
        """
        Initializes a new database session.

        Args:
            recreate: Recreates the database tables if True
        """

        # Table name
        table = self.setting("table", "vectors")

        # Create vectors table
        self.table = Table(
            table,
            MetaData(),
            Column("indexid", Integer, primary_key=True, autoincrement=False),
            Column("embedding", Vector(self.config["dimensions"])),
        )

        # Create ANN index - inner product is equal to cosine similarity on normalized vectors
        index = Index(
            f"{table}-index",
            self.table.c["embedding"],
            postgresql_using="hnsw",
            postgresql_with=self.settings(),
            postgresql_ops={"embedding": "vector_ip_ops"},
        )

        # Drop and recreate table
        if recreate:
            self.table.drop(self.engine, checkfirst=True)
            index.drop(self.engine, checkfirst=True)

        # Create table and index
        self.table.create(self.engine, checkfirst=True)
        index.create(self.engine, checkfirst=True)

    def settings(self):
        """
        Returns settings for this index.

        Returns:
            dict
        """

        return {"m": self.setting("m", 16), "ef_construction": self.setting("efconstruction", 200)}
