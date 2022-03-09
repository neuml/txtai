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

    INSERT_DOCUMENT = "REPLACE INTO documents VALUES (?, ?, ?, ?)"
    DELETE_DOCUMENTS = "DELETE FROM documents WHERE id IN (SELECT id FROM batch)"

    # Objects - stores binary content
    CREATE_OBJECTS = """
        CREATE TABLE IF NOT EXISTS objects (
            id TEXT PRIMARY KEY,
            object BLOB,
            tags TEXT,
            entry DATETIME
        )
    """

    INSERT_OBJECT = "REPLACE INTO objects VALUES (?, ?, ?, ?)"
    DELETE_OBJECTS = "DELETE FROM objects WHERE id IN (SELECT id FROM batch)"

    # Sections - stores section text
    CREATE_SECTIONS = """
        CREATE TABLE IF NOT EXISTS %s (
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
    COPY_SECTIONS = (
        "INSERT INTO %s SELECT (select count(*) - 1 from sections s1 where s.indexid >= s1.indexid) indexid, "
        + "s.id, %s AS text, s.tags, s.entry FROM sections s LEFT JOIN documents d ON s.id = d.id ORDER BY indexid"
    )
    STREAM_SECTIONS = "SELECT s.id, s.text, object, s.tags FROM %s s LEFT JOIN objects o ON s.id = o.id ORDER BY indexid"
    DROP_SECTIONS = "DROP TABLE sections"
    RENAME_SECTIONS = "ALTER TABLE %s RENAME TO sections"

    # Queries
    SELECT_IDS = "SELECT indexid, id FROM sections WHERE id in (SELECT id FROM batch)"

    # Partial sql clauses
    TABLE_CLAUSE = (
        "SELECT %s FROM sections s "
        + "LEFT JOIN documents d ON s.id = d.id "
        + "LEFT JOIN objects o ON s.id = o.id "
        + "LEFT JOIN scores sc ON s.indexid = sc.indexid"
    )
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

        # Register custom functions
        self.addfunctions()

    def insert(self, documents, index=0):
        # Initialize connection if not open
        self.initialize()

        # Get entry date
        entry = datetime.datetime.now()

        # Insert documents
        for uid, document, tags in documents:
            if isinstance(document, dict):
                # Insert document and use return value for sections table
                document = self.insertdocument(uid, document, tags, entry)

            if document is not None:
                if isinstance(document, list):
                    # Join tokens to text
                    document = " ".join(document)
                elif not isinstance(document, str):
                    # If object support is enabled, save object
                    self.insertobject(uid, document, tags, entry)

                    # Clear section text for objects, even when objects aren't inserted
                    document = None

                # Save text section
                self.insertsection(index, uid, document, tags, entry)
                index += 1

    def delete(self, ids):
        if self.connection:
            # Batch ids
            self.batch(ids=ids)

            # Delete all documents, objects and sections by id
            self.cursor.execute(SQLite.DELETE_DOCUMENTS)
            self.cursor.execute(SQLite.DELETE_OBJECTS)
            self.cursor.execute(SQLite.DELETE_SECTIONS)

    def reindex(self, columns=None):
        if self.connection:
            # Working table name
            name = "rebuild"

            # Resolve and build column strings if provided
            select = "text"
            if columns:
                select = "|| ' ' ||".join([self.resolve(c) for c in columns])

            # Create new table to hold reordered sections
            self.cursor.execute(SQLite.CREATE_SECTIONS % name)

            # Copy data over
            self.cursor.execute(SQLite.COPY_SECTIONS % (name, select))

            # Stream new results
            self.cursor.execute(SQLite.STREAM_SECTIONS % name)
            for uid, text, obj, tags in self.cursor:
                if not text and self.encoder and obj:
                    yield (uid, self.encoder.decode(obj), tags)
                else:
                    yield (uid, text, tags)

            # Swap as new table
            self.cursor.execute(SQLite.DROP_SECTIONS)
            self.cursor.execute(SQLite.RENAME_SECTIONS % name)
            self.cursor.execute(SQLite.CREATE_SECTIONS_INDEX)

    def save(self, path):
        # Temporary database
        if not self.path:
            # Save temporary database
            self.connection.commit()

            # Copy data from current to new
            connection = self.copy(path)

            # Close temporary database
            self.connection.close()

            # Point connection to new connection
            self.connection = connection
            self.cursor = self.connection.cursor()
            self.path = path

            # Register custom functions
            self.addfunctions()

        # Paths are equal, commit changes
        elif self.path == path:
            self.connection.commit()

        # New path is different from current path, copy data and continue using current connection
        else:
            self.copy(path).close()

    def close(self):
        # Close connection
        if self.connection:
            self.connection.close()

    def ids(self, ids):
        # Batch ids and run query
        self.batch(ids=ids)
        self.cursor.execute(SQLite.SELECT_IDS)

        # Format and return results
        return self.cursor.fetchall()

    def resolve(self, name, alias=None):
        # Standard column names
        sections = ["indexid", "id", "tags", "entry"]
        noprefix = ["data", "object", "score", "text"]

        # Alias expression
        if alias:
            # Skip if name matches alias or alias is a standard column name
            if name == alias or alias in sections:
                return name

            # Build alias clause
            return f'{name} as "{alias}"'

        # Name is already resolved, skip
        if name.startswith("json_extract(data") or any(f"s.{s}" == name for s in sections):
            return name

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
                    # Decode object
                    if self.encoder and column == "object":
                        result[column] = self.encoder.decode(row[x])
                    else:
                        result[column] = row[x]

            results.append(result)

        return results

    def addfunctions(self):
        """
        Adds custom functions in current connection.
        """

        if self.connection and self.functions:
            # Enable callback tracebacks to show user-defined function errors
            sqlite3.enable_callback_tracebacks(True)

            for name, argcount, fn in self.functions:
                self.connection.create_function(name, argcount, fn)

    def initialize(self):
        """
        Creates connection and initial database schema if no connection exists.
        """

        if not self.connection:
            name = "sections"

            # Create temporary database. Thread locking must be handled externally.
            self.connection = sqlite3.connect("", check_same_thread=False)
            self.cursor = self.connection.cursor()

            # Register custom functions
            self.addfunctions()

            # Create initial schema and indices
            self.cursor.execute(SQLite.CREATE_DOCUMENTS)
            self.cursor.execute(SQLite.CREATE_OBJECTS)
            self.cursor.execute(SQLite.CREATE_SECTIONS % name)
            self.cursor.execute(SQLite.CREATE_SECTIONS_INDEX)

    def insertdocument(self, uid, document, tags, entry):
        """
        Inserts a document.

        Args:
            uid: unique id
            document: input document
            tags: document tags
            entry: generated entry date

        Returns:
            section value
        """

        # Make a copy of document before changing
        document = document.copy()

        # Get and remove object field from document
        obj = document.pop("object") if "object" in document else None

        # Insert document as JSON
        if document:
            self.cursor.execute(SQLite.INSERT_DOCUMENT, [uid, json.dumps(document, allow_nan=False), tags, entry])

        # Get value of text field
        text = document.get("text")

        # If both text and object are set, insert object as it won't otherwise be used
        if text and obj:
            self.insertobject(uid, obj, tags, entry)

        # Return value to use for section - use text if available otherwise use object
        return text if text else obj

    def insertobject(self, uid, obj, tags, entry):
        """
        Inserts an object.

        Args:
            uid: unique id
            obj: input object
            tags: object tags
            entry: generated entry date
        """

        # If object support is enabled, save object
        if self.encoder:
            self.cursor.execute(SQLite.INSERT_OBJECT, [uid, self.encoder.encode(obj), tags, entry])

    def insertsection(self, index, uid, text, tags, entry):
        """
        Inserts a section.

        Args:
            index: index id
            uid: unique id
            text: section text
            tags: section tags
            entry: generated entry date
        """

        # Save text section
        self.cursor.execute(SQLite.INSERT_SECTION, [index, uid, text, tags, entry])

    def copy(self, path):
        """
        Copies the current database into path. This method will use the backup API if the current connection has no uncommitted changes.
        Otherwise, iterdump is used, as the backup call will hang for an uncommitted connection.

        Args:
            path: path to write database

        Returns:
            new connection with data copied over
        """

        # Delete existing file, if necessary
        if os.path.exists(path):
            os.remove(path)

        # Create database. Thread locking must be handled externally.
        connection = sqlite3.connect(path, check_same_thread=False)

        if self.connection.in_transaction:
            # The backup call will hang if there are uncommitted changes, need to copy over
            # with iterdump (which is much slower)
            for sql in self.connection.iterdump():
                connection.execute(sql)
        else:
            # Database is up to date, can do a more efficient copy with SQLite C API
            self.connection.backup(connection)

        return connection

    def defaults(self):
        """
        Returns a list of default columns when there is no select clause.

        Returns:
            list of default columns
        """

        return "s.id, text, score"

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
