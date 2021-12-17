"""
SQLite module
"""

import datetime
import json
import os
import sqlite3

from .base import Database


class SQLite(Database):
    """
    Document database backed by SQLite.
    """

    # Temporary table for working with id batches
    CREATE_BATCH = """
        CREATE TEMP TABLE IF NOT EXISTS batch (
            indexid INTEGER,
            id TEXT,
            batch INTEGER
        )
    """

    DELETE_BATCH = "DELETE FROM batch"
    INSERT_BATCH = "INSERT INTO batch VALUES (?, ?, ?)"

    # Temporary table for joining similarity scores
    CREATE_SCORES = """
        CREATE TEMP TABLE IF NOT EXISTS scores (
            indexid INTEGER,
            score REAL
        )
    """

    DELETE_SCORES = "DELETE FROM scores"
    INSERT_SCORE = "INSERT INTO scores VALUES (?, ?)"

    # Documents - stores full content
    CREATE_DOCUMENTS = """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            data JSON,
            tags TEXT,
            entry DATETIME
        )
    """

    INSERT_DOCUMENT = "INSERT INTO documents VALUES (?, ?, ?, ?)"
    DELETE_DOCUMENTS = "DELETE FROM documents WHERE id IN (SELECT id FROM batch)"

    # Sections - stores section text
    CREATE_SECTIONS = """
        CREATE TABLE IF NOT EXISTS sections (
            indexid INTEGER PRIMARY KEY,
            id TEXT,
            text TEXT,
            tags TEXT,
            entry DATETIME
        )
    """

    CREATE_SECTIONS_INDEX = "CREATE INDEX section_id ON sections(id)"
    INSERT_SECTION = "INSERT INTO sections VALUES (?, ?, ?, ?, ?)"
    DELETE_SECTIONS = "DELETE FROM sections WHERE id IN (SELECT id FROM batch)"

    # Queries
    SELECT_IDS = "SELECT indexid, id FROM sections WHERE id in (SELECT id FROM batch)"

    # Partial sql clauses
    TABLE_CLAUSE = "SELECT %s FROM sections s LEFT JOIN documents d ON s.id = d.id LEFT JOIN scores sc on s.indexid=sc.indexid"
    IDS_CLAUSE = "s.indexid in (SELECT indexid from batch WHERE batch=%s)"

    def __init__(self, config):
        """
        Creates a new Database.

        Args:
            config: database configuration parameters
        """

        super().__init__(config)

        # SQLite connection handle
        self.connection = None
        self.cursor = None
        self.path = None

    def load(self, path):
        # Load an existing database. Thread locking must be handled externally.
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.cursor = self.connection.cursor()
        self.path = path

    def insert(self, documents, index=0):
        # Initialize connection if not open
        self.initialize()

        # Get entry date
        entry = datetime.datetime.now()

        # Insert documents
        for uid, document, tags in documents:
            if isinstance(document, dict):
                # Insert document as JSON
                self.cursor.execute(SQLite.INSERT_DOCUMENT, [uid, json.dumps(document), tags, entry])

                # Extract text, if any
                document = document.get("text")

            if isinstance(document, list):
                # Join tokens to text
                document = " ".join(document)

            if document:
                # Save text section
                self.cursor.execute(SQLite.INSERT_SECTION, [index, uid, document, tags, entry])
                index += 1

    def delete(self, ids):
        if self.connection:
            # Batch ids
            self.batch(ids=ids)

            # Delete all documents and sections by id
            self.cursor.execute(SQLite.DELETE_DOCUMENTS)
            self.cursor.execute(SQLite.DELETE_SECTIONS)

    def save(self, path):
        # Save the changes
        self.connection.commit()

        # If this is a temporary database, copy over to database at path
        if not self.path:
            # Delete existing file, if necessary
            if os.path.exists(path):
                os.remove(path)

            # Create database. Thread locking must be handled externally.
            connection = sqlite3.connect(path, check_same_thread=False)

            # Copy from existing database to new database
            self.connection.backup(connection)
            self.connection.close()

            # Point connection to new database
            self.connection = connection
            self.cursor = self.connection.cursor()
            self.path = path

    def ids(self, ids):
        # Batch ids and run query
        self.batch(ids=ids)
        self.cursor.execute(SQLite.SELECT_IDS)

        # Format and return results
        return self.cursor.fetchall()

    def query(self, query, limit):
        # Extract query components
        select = query.get("select", self.defaults())
        where = query.get("where")
        groupby, having = query.get("groupby"), query.get("having")
        orderby, qlimit = query.get("orderby"), query.get("limit")
        similarity = query.get("similar")

        # Build query text
        query = SQLite.TABLE_CLAUSE % select
        if where is not None:
            query += f" WHERE {where}"
        if groupby is not None:
            query += f" GROUP BY {groupby}"
        if having is not None:
            query += f" HAVING {having}"
        if orderby is not None:
            query += f" ORDER BY {orderby}"

        # Default ORDER BY if not provided and similarity scores are available
        if similarity and orderby is None:
            query += " ORDER BY score DESC"

        # Apply query limit
        if qlimit is not None or limit:
            query += f" LIMIT {qlimit if qlimit else limit}"

        # Clear scores when no similar clauses present
        if not similarity:
            self.scores(None)

        # Runs a user query through execute method, which has common user query handling logic
        self.execute(self.cursor.execute, query)

        # Retrieve column list from query
        columns = [c[0] for c in self.cursor.description]

        # Map results and return
        results = []
        for row in self.cursor:
            result = {}

            # Copy columns to result. In cases with duplicate column names, find one with a value
            for x, column in enumerate(columns):
                if column not in result or result[column] is None:
                    result[column] = row[x]

            results.append(result)

        return results

    def resolve(self, name, alias=False, compound=False):
        # Standard column names
        sections = ["indexid", "id", "tags", "entry"]
        noprefix = ["data", "score", "text"]

        # Alias JSON column expressions
        if alias:
            # Only apply aliases to non-standard columns or compound expressions
            if name not in sections + noprefix or compound:
                return f' as "{name}"'

            # No alias
            return None

        # Standard columns - need prefixes
        if name.lower() in sections:
            return f"s.{name}"

        # Standard columns - no prefixes
        if name.lower() in noprefix:
            return name

        # Other columns come from documents.data JSON
        return f'json_extract(data, "$.{name}")'

    def embed(self, similarity, batch):
        # Load similarity results id batch
        self.batch(indexids=[i for i, _ in similarity[batch]], batch=batch)

        # Average and load all similarity scores with first batch
        if not batch:
            self.scores(similarity)

        # Return ids clause placeholder
        return SQLite.IDS_CLAUSE % batch

    def defaults(self):
        """
        Returns a list of default columns when there is no select clause.

        Returns:
            list of default columns
        """

        return "s.id, text, score"

    def initialize(self):
        """
        Creates connection and initial database schema if no connection exists.
        """

        if not self.connection:
            # Create temporary database. Thread locking must be handled externally.
            self.connection = sqlite3.connect("", check_same_thread=False)
            self.cursor = self.connection.cursor()

            # Create initial schema and indices
            self.cursor.execute(SQLite.CREATE_DOCUMENTS)
            self.cursor.execute(SQLite.CREATE_SECTIONS)
            self.cursor.execute(SQLite.CREATE_SECTIONS_INDEX)

    def batch(self, indexids=None, ids=None, batch=None):
        """
        Loads ids to a temporary batch table for efficient query processing.

        Args:
            indexids: list of indexids
            ids: list of ids
            batch: batch index, used when statement has multiple subselects
        """

        # Create or Replace temporary batch table
        self.cursor.execute(SQLite.CREATE_BATCH)

        # Delete batch when batch id is empty or for batch 0
        if not batch:
            self.cursor.execute(SQLite.DELETE_BATCH)

        if indexids:
            self.cursor.executemany(SQLite.INSERT_BATCH, [(i, None, batch) for i in indexids])
        if ids:
            self.cursor.executemany(SQLite.INSERT_BATCH, [(None, str(uid), batch) for uid in ids])

    def scores(self, similarity):
        """
        Loads a batch of similarity scores to a temporary table for efficient query processing.

        Args:
            similarity: similarity results as [(indexid, score)]
        """

        # Create or Replace temporary scores table
        self.cursor.execute(SQLite.CREATE_SCORES)

        # Delete scores
        self.cursor.execute(SQLite.DELETE_SCORES)

        if similarity:
            # Average scores per id, needed for multiple similar() clauses
            scores = {}
            for s in similarity:
                for i, score in s:
                    if i not in scores:
                        scores[i] = []
                    scores[i].append(score)

            # Average scores by id
            self.cursor.executemany(SQLite.INSERT_SCORE, [(i, sum(s) / len(s)) for i, s in scores.items()])
