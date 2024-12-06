"""
SQLite module
"""

import os
import sqlite3

# Conditional import
try:
    import sqlite_vec

    SQLITEVEC = True
except ImportError:
    SQLITEVEC = False

from .base import ANN


class SQLite(ANN):
    """
    Builds an ANN index backed by a SQLite database.
    """

    def __init__(self, config):
        super().__init__(config)

        if not SQLITEVEC:
            raise ImportError('sqlite-vec is not available - install "ann" extra to enable')

        # Database parameters
        self.connection, self.cursor, self.path = None, None, ""

        # Quantization setting
        self.quantize = self.setting("quantize")
        self.quantize = 8 if isinstance(self.quantize, bool) else int(self.quantize) if self.quantize else None

    def load(self, path):
        self.path = path

    def index(self, embeddings):
        # Initialize tables
        self.initialize(recreate=True)

        # Add vectors
        self.database().executemany(self.insertsql(), enumerate(embeddings))

        # Add id offset and index build metadata
        self.config["offset"] = embeddings.shape[0]
        self.metadata(self.settings())

    def append(self, embeddings):
        self.database().executemany(self.insertsql(), [(x + self.config["offset"], row) for x, row in enumerate(embeddings)])

        self.config["offset"] += embeddings.shape[0]
        self.metadata()

    def delete(self, ids):
        self.database().executemany(self.deletesql(), [(x,) for x in ids])

    def search(self, queries, limit):
        results = []
        for query in queries:
            # Execute query
            self.database().execute(self.searchsql(), [query, limit])

            # Add query results
            results.append(list(self.database()))

        return results

    def count(self):
        self.database().execute(self.countsql())
        return self.cursor.fetchone()[0]

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

        # Paths are equal, commit changes
        elif self.path == path:
            self.connection.commit()

        # New path is different from current path, copy data and continue using current connection
        else:
            self.copy(path).close()

    def close(self):
        # Parent logic
        super().close()

        # Close database connection
        if self.connection:
            self.connection.close()
            self.connection = None

    def initialize(self, recreate=False):
        """
        Initializes a new database session.

        Args:
            recreate: Recreates the database tables if True
        """

        # Create table
        self.database().execute(self.tablesql())

        # Clear data
        if recreate:
            self.database().execute(self.tosql("DELETE FROM {table}"))

    def settings(self):
        """
        Returns settings for this index.

        Returns:
            dict
        """

        sqlite, sqlitevec = self.database().execute("SELECT sqlite_version(), vec_version()").fetchone()

        return {"sqlite": sqlite, "sqlite-vec": sqlitevec}

    def database(self):
        """
        Gets the current database cursor. Creates a new connection
        if there isn't one.

        Returns:
            cursor
        """

        if not self.connection:
            self.connection = self.connect(self.path)
            self.cursor = self.connection.cursor()

        return self.cursor

    def connect(self, path):
        """
        Creates a new database connection.

        Args:
            path: path to database file

        Returns:
            database connection
        """

        # Create connection
        connection = sqlite3.connect(path, check_same_thread=False)

        # Load sqlite-vec extension
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)

        # Return connection and cursor
        return connection

    def copy(self, path):
        """
        Copies content from the current database into target.

        Args:
            path: target database path

        Returns:
            new database connection
        """

        # Delete existing file, if necessary
        if os.path.exists(path):
            os.remove(path)

        # Create new connection
        connection = self.connect(path)

        if self.connection.in_transaction:
            # Initialize connection
            connection.execute(self.tablesql())

            # The backup call will hang if there are uncommitted changes, need to copy over
            # with iterdump (which is much slower)
            for sql in self.connection.iterdump():
                if self.tosql('insert into "{table}"') in sql.lower():
                    connection.execute(sql)
        else:
            # Database is up to date, can do a more efficient copy with SQLite C API
            self.connection.backup(connection)

        return connection

    def tablesql(self):
        """
        Builds a CREATE table statement for table.

        Returns:
            CREATE TABLE
        """

        # Binary quantization
        if self.quantize == 1:
            embedding = f"embedding BIT[{self.config['dimensions']}]"

        # INT8 quantization
        elif self.quantize == 8:
            embedding = f"embedding INT8[{self.config['dimensions']}] distance=cosine"

        # Standard FLOAT32
        else:
            embedding = f"embedding FLOAT[{self.config['dimensions']}] distance=cosine"

        # Return CREATE TABLE sql
        return self.tosql(("CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0" "(indexid INTEGER PRIMARY KEY, " f"{embedding})"))

    def insertsql(self):
        """
        Creates an INSERT SQL statement.

        Returns:
            INSERT
        """

        return self.tosql(f"INSERT INTO {{table}}(indexid, embedding) VALUES (?, {self.embeddingsql()})")

    def deletesql(self):
        """
        Creates a DELETE SQL statement.

        Returns:
            DELETE
        """

        return self.tosql("DELETE FROM {table} WHERE indexid = ?")

    def searchsql(self):
        """
        Creates a SELECT SQL statement for search.

        Returns:
            SELECT
        """

        return self.tosql(("SELECT indexid, 1 - distance FROM {table} " f"WHERE embedding MATCH {self.embeddingsql()} AND k = ? ORDER BY distance"))

    def countsql(self):
        """
        Creates a SELECT COUNT statement.

        Returns:
            SELECT COUNT
        """

        return self.tosql("SELECT count(indexid) FROM {table}")

    def embeddingsql(self):
        """
        Creates an embeddings column SQL snippet.

        Returns:
            embeddings column SQL
        """

        # Binary quantization
        if self.quantize == 1:
            embedding = "vec_quantize_binary(?)"

        # INT8 quantization
        elif self.quantize == 8:
            embedding = "vec_quantize_int8(?, 'unit')"

        # Standard FLOAT32
        else:
            embedding = "?"

        return embedding

    def tosql(self, sql):
        """
        Creates a SQL statement substituting in the configured table name.

        Args:
            sql: SQL statement with a {table} parameter

        Returns:
            fully resolved SQL statement
        """

        table = self.setting("table", "vectors")
        return sql.format(table=table)
