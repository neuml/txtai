"""
DuckDB module
"""

import os

from tempfile import TemporaryDirectory

# Conditional import
try:
    import duckdb

    DUCKDB = True
except ImportError:
    DUCKDB = False

from .filedb import FileDB


class DuckDB(FileDB):
    """
    Database instance backed by DuckDB.
    """

    # Delete single document and object
    DELETE_DOCUMENT = "DELETE FROM documents WHERE id = ?"
    DELETE_OBJECT = "DELETE FROM objects WHERE id = ?"

    def __init__(self, config):
        super().__init__(config)

        if not DUCKDB:
            raise ImportError('DuckDB is not available - install "database" extra to enable')

    def insertdocument(self, uid, document, tags, entry):
        if document:
            # Delete existing document
            self.cursor.execute(DuckDB.DELETE_DOCUMENT, [uid])

        # Call parent logic
        return super().insertdocument(uid, document, tags, entry)

    def insertobject(self, uid, obj, tags, entry):
        if self.encoder:
            # Delete existing object
            self.cursor.execute(DuckDB.DELETE_OBJECT, [uid])

        # Call parent logic
        super().insertobject(uid, obj, tags, entry)

    def connect(self, path=":memory:"):
        # Create connection and start a transaction
        # pylint: disable=I1101
        connection = duckdb.connect(path)
        connection.begin()

        return connection

    def getcursor(self):
        return self.connection

    def rows(self):
        # Iteratively retrieve and yield rows
        batch = 256
        rows = self.cursor.fetchmany(batch)
        while rows:
            yield from rows
            rows = self.cursor.fetchmany(batch)

    def addfunctions(self):
        # DuckDB doesn't currently support scalar functions
        return

    def copy(self, path):
        # Delete existing file, if necessary
        if os.path.exists(path):
            os.remove(path)

        # Create database connection
        # pylint: disable=I1101
        connection = duckdb.connect(path)

        # List of tables
        tables = ["documents", "objects", "sections"]

        with TemporaryDirectory() as directory:
            # Export existing tables
            for table in tables:
                self.connection.execute(f"COPY {table} TO '{directory}/{table}.parquet' (FORMAT parquet)")

            # Create initial schema
            for schema in [FileDB.CREATE_DOCUMENTS, FileDB.CREATE_OBJECTS, FileDB.CREATE_SECTIONS % "sections"]:
                connection.execute(schema)

            # Import tables into new schema
            for table in tables:
                connection.execute(f"COPY {table} FROM '{directory}/{table}.parquet' (FORMAT parquet)")

            # Create indexes and sync data to database file
            connection.execute(FileDB.CREATE_SECTIONS_INDEX)
            connection.execute("CHECKPOINT")

        # Start transaction
        connection.begin()

        return connection
