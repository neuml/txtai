"""
Client module
"""

import os
import time

# Conditional import
try:
    from sqlalchemy import Text, cast, create_engine, event, insert, text as textsql
    from sqlalchemy.orm import Session, aliased, scoped_session, sessionmaker
    from sqlalchemy.schema import CreateSchema, CreateTable

    from .schema import Base, Batch, Document, Object, Section, SectionBase, Score

    ORM = True
except ImportError:
    ORM = False

from .rdbms import RDBMS


class Client(RDBMS):
    """
    Database client instance. This class connects to an external database using SQLAlchemy. It supports any database
    that is supported by SQLAlchemy (PostgreSQL, MariaDB, etc) and has JSON support.
    """

    def __init__(self, config):
        """
        Creates a new Database.

        Args:
            config: database configuration parameters
        """

        super().__init__(config)

        if not ORM:
            raise ImportError('SQLAlchemy is not available - install "database" extra to enable')

        # SQLAlchemy engine
        self.engine = None

    def save(self, path):
        # Commit session
        super().save(path)

    def close(self):
        # Remove all thread-local Sessions from the scoped_session registry, then dispose the engine.
        if self.connection:
            self.connection.remove()
            self.connection = None
        if self.engine:
            self.engine.dispose()
            self.engine = None

    def reindexstart(self):
        # Working table name
        name = f"rebuild{round(time.time() * 1000)}"

        # Create working table via the current session (not engine.begin()) so the DDL
        # shares the session's existing transaction.  Opening a second engine connection
        # while the session already holds a write lock would deadlock on SQLite.
        type("Rebuild", (SectionBase,), {"__tablename__": name})
        self.connection.execute(CreateTable(Base.metadata.tables[name]))

        return name

    def reindexend(self, name):
        # Remove table object from metadata
        Base.metadata.remove(Base.metadata.tables[name])

    def jsonprefix(self):
        # JSON column prefix
        return "cast("

    def jsoncolumn(self, name):
        # Alias documents table
        d = aliased(Document, name="d")

        # Build JSON column expression for column
        return str(cast(d.data[name].as_string(), Text).compile(dialect=self.engine.dialect, compile_kwargs={"literal_binds": True}))

    def createtables(self):
        # Create persistent tables via engine so DDL auto-commits and is visible to all sessions.
        Base.metadata.create_all(self.engine, checkfirst=True)

        # Clear existing data - table schema is created upon connecting to database
        for table in ["sections", "documents", "objects"]:
            self.cursor.execute(f"DELETE FROM {table}")

    def finalize(self):
        # Flush cached objects
        self.connection.flush()

    def insertdocument(self, uid, data, tags, entry):
        self.connection.add(Document(id=uid, data=data, tags=tags, entry=entry))

    def insertobject(self, uid, data, tags, entry):
        self.connection.add(Object(id=uid, object=data, tags=tags, entry=entry))

    def insertsection(self, index, uid, text, tags, entry):
        # Save text section
        self.connection.add(Section(indexid=index, id=uid, text=text, tags=tags, entry=entry))

    def createbatch(self):
        # Temporary batch/scores tables are created per-connection in _init_temp_tables()
        # (the engine's connect event). No explicit DDL is needed here.
        pass

    def insertbatch(self, indexids, ids, batch):
        if indexids:
            self.connection.execute(insert(Batch), [{"indexid": i, "batch": batch} for i in indexids])
        if ids:
            self.connection.execute(insert(Batch), [{"id": str(uid), "batch": batch} for uid in ids])

    def createscores(self):
        # Temporary batch/scores tables are created per-connection in _init_temp_tables()
        # (the engine's connect event). No explicit DDL is needed here.
        pass

    def insertscores(self, scores):
        # Average scores by id
        if scores:
            self.connection.execute(insert(Score), [{"indexid": i, "score": sum(s) / len(s)} for i, s in scores.items()])

    def connect(self, path=None):
        # Connection URL
        content = self.config.get("content")

        # Read ENV variable, if necessary
        content = os.environ.get("CLIENT_URL") if content == "client" else content

        # Create engine with the default QueuePool so connections are reused across operations.
        self.engine = create_engine(content, echo=False, json_serializer=lambda x: x)

        # Register a connect listener that initialises per-connection TEMP tables.
        # Temporary tables (batch, scores) are connection-scoped in all major SQL databases,
        # so they must be created on every new physical connection rather than once via
        # engine.begin() (which would create-and-immediately-discard them).  The connect event
        # fires once per physical connection — with QueuePool the connection is reused by the
        # same thread across operations, so the TEMP tables persist for the session's lifetime.
        event.listen(self.engine, "connect", Client._init_temp_tables)

        # Create a per-thread scoped session factory.  scoped_session proxies all Session
        # calls to a thread-local Session instance, eliminating the shared-Session race that
        # caused 'prepared'-state corruption when concurrent requests hit the same Session.
        database = scoped_session(sessionmaker(bind=self.engine))

        # Set default schema, if necessary
        schema = self.config.get("schema")
        if schema:
            with self.engine.begin() as conn:
                if conn.dialect.name == "postgresql":
                    conn.execute(CreateSchema(schema, if_not_exists=True))

            if self.engine.dialect.name == "postgresql":
                database.execute(textsql("SET search_path TO :schema"), {"schema": schema})

        return database

    def getcursor(self):
        return Cursor(self.connection)

    def rows(self):
        return self.cursor

    def addfunctions(self):
        return

    def sqldialect(self, database, sql, parameters=None):
        """
        Executes a SQL statement based on the current SQL dialect.

        Args:
            database: current database
            sql: SQL to execute
            parameters: optional bind parameters
        """

        args = (sql, parameters) if self.engine.dialect.name == "postgresql" else (textsql("SELECT 1"),)
        database.execute(*args)

    @staticmethod
    def _init_temp_tables(dbapi_connection, connection_record):
        """
        Creates per-connection temporary tables for batch and score operations.

        Temporary tables are connection-scoped in all major SQL databases (SQLite, PostgreSQL,
        MariaDB).  This connect event fires once per physical connection so each pooled
        connection gets its own isolated batch/scores workspace without cross-thread
        interference.

        Args:
            dbapi_connection: raw DBAPI connection
            connection_record: connection pool record (unused)
        """

        cursor = dbapi_connection.cursor()

        # Batch table — temporary workspace for id lookups during search
        cursor.execute(
            """
            CREATE TEMPORARY TABLE IF NOT EXISTS batch (
                indexid INTEGER,
                id      TEXT,
                batch   INTEGER
            )
            """
        )

        # Scores table — temporary workspace for similarity score joins
        cursor.execute(
            """
            CREATE TEMPORARY TABLE IF NOT EXISTS scores (
                indexid INTEGER PRIMARY KEY,
                score   REAL
            )
            """
        )

        cursor.close()


class Cursor:
    """
    Implements basic compatibility with the Python DB-API.
    """

    def __init__(self, connection):
        self.connection = connection
        self.result = None

    def __iter__(self):
        return self.result

    def execute(self, statement, parameters=None):
        """
        Executes statement.

        Args:
            statement: statement to execute
            parameters: optional dictionary with bind parameters
        """

        if isinstance(statement, str):
            statement = textsql(statement)

        self.result = self.connection.execute(statement, parameters)

    def fetchall(self):
        """
        Fetches all rows from the current result.

        Returns:
            all rows from current result
        """

        return self.result.all() if self.result else None

    def fetchone(self):
        """
        Fetches first row from current result.

        Returns:
            first row from current result
        """

        return self.result.first() if self.result else None

    @property
    def description(self):
        """
        Returns columns for current result.

        Returns:
            list of columns
        """

        return [(key,) for key in self.result.keys()] if self.result else None
