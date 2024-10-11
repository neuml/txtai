"""
PGText module
"""

import os

# Conditional import
try:
    from sqlalchemy import create_engine, desc, delete, text
    from sqlalchemy import Column, Computed, Index, Integer, MetaData, StaticPool, Table, Text
    from sqlalchemy.dialects.postgresql import TSVECTOR
    from sqlalchemy.orm import Session

    PGTEXT = True
except ImportError:
    PGTEXT = False

from .base import Scoring


class PGText(Scoring):
    """
    Postgres full text search (FTS) based scoring.
    """

    def __init__(self, config=None):
        super().__init__(config)

        if not PGTEXT:
            raise ImportError('PGText is not available - install "scoring" extra to enable')

        # Database connection
        self.engine, self.database, self.table = None, None, None

        # Language
        self.language = self.config.get("language", "english")

    def insert(self, documents, index=None):
        # Initialize tables
        self.initialize(recreate=True)

        # Collection of rows to insert
        rows = []

        # Collect rows
        for uid, document, _ in documents:
            # Extract text, if necessary
            if isinstance(document, dict):
                document = document.get(self.text, document.get(self.object))

            if document is not None:
                # If index is passed, use indexid, otherwise use id
                uid = index if index is not None else uid

                # Add row if the data type is accepted
                if isinstance(document, (str, list)):
                    rows.append((uid, " ".join(document) if isinstance(document, list) else document))

                # Increment index
                index = index + 1 if index is not None else None

        # Insert rows
        self.database.execute(self.table.insert(), [{"indexid": x, "text": text} for x, text in rows])

    def delete(self, ids):
        self.database.execute(delete(self.table).where(self.table.c["indexid"].in_(ids)))

    def weights(self, tokens):
        # Not supported
        return None

    def search(self, query, limit=3):
        # Run query
        query = (
            self.database.query(self.table.c["indexid"], text("ts_rank(vector, plainto_tsquery(:language, :query)) rank"))
            .order_by(desc(text("rank")))
            .limit(limit)
            .params({"language": self.language, "query": query})
        )

        return [(uid, score) for uid, score in query if score > 1e-5]

    def batchsearch(self, queries, limit=3, threads=True):
        return [self.search(query, limit) for query in queries]

    def count(self):
        return self.database.query(self.table.c["indexid"]).count()

    def load(self, path):
        # Reset database to original checkpoint
        self.database.rollback()

        # Initialize tables
        self.initialize()

    def save(self, path):
        self.database.commit()

    def close(self):
        self.database.close()

    def hasterms(self):
        return True

    def isnormalized(self):
        return True

    def initialize(self, recreate=False):
        """
        Initializes a new database session.

        Args:
            recreate: Recreates the database tables if True
        """

        if not self.database:
            # Create engine and session
            self.engine = create_engine(self.config.get("url", os.environ.get("SCORING_URL")), poolclass=StaticPool, echo=False)
            self.database = Session(self.engine)

            # Table name
            table = self.config.get("table", "scoring")

            # Create vectors table
            self.table = Table(
                table,
                MetaData(),
                Column("indexid", Integer, primary_key=True, autoincrement=False),
                Column("text", Text),
                (
                    Column("vector", TSVECTOR, Computed(f"to_tsvector('{self.language}', text)", persisted=True))
                    if self.engine.dialect.name == "postgresql"
                    else Column("vector", Integer)
                ),
            )

            # Create ANN index - inner product is equal to cosine similarity on normalized vectors
            index = Index(
                f"{table}-index",
                self.table.c["vector"],
                postgresql_using="gin",
            )

            # Drop and recreate table
            if recreate:
                self.table.drop(self.engine, checkfirst=True)
                index.drop(self.engine, checkfirst=True)

            # Create table and index
            self.table.create(self.engine, checkfirst=True)
            index.create(self.engine, checkfirst=True)
