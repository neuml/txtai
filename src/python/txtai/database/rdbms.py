"""
RDBMS module
"""

import datetime
import json

from .base import Database
from .schema import Statement


# pylint: disable=R0904
class RDBMS(Database):
    """
    Base relational database class. A relational database uses SQL to insert, update, delete and select from a
    database instance.
    """

    def __init__(self, config):
        """
        Creates a new Database.

        Args:
            config: database configuration parameters
        """

        super().__init__(config)

        # Database connection
        self.connection = None
        self.cursor = None

    def load(self, path):
        # Load an existing database. Thread locking must be handled externally.
        self.session(path)

    def insert(self, documents, index=0):
        # Initialize connection if not open
        self.initialize()

        # Get entry date
        entry = datetime.datetime.now(datetime.timezone.utc)

        # Insert documents
        for uid, document, tags in documents:
            if isinstance(document, dict):
                # Insert document and use return value for sections table
                document = self.loaddocument(uid, document, tags, entry)

            if document is not None:
                if isinstance(document, list):
                    # Join tokens to text
                    document = " ".join(document)
                elif not isinstance(document, str):
                    # If object support is enabled, save object
                    self.loadobject(uid, document, tags, entry)

                    # Clear section text for objects, even when objects aren't inserted
                    document = None

                # Save text section
                self.loadsection(index, uid, document, tags, entry)
                index += 1

        # Post processing logic
        self.finalize()

    def delete(self, ids):
        if self.connection:
            # Batch ids
            self.batch(ids=ids)

            # Delete all documents, objects and sections by id
            self.cursor.execute(Statement.DELETE_DOCUMENTS)
            self.cursor.execute(Statement.DELETE_OBJECTS)
            self.cursor.execute(Statement.DELETE_SECTIONS)

    def reindex(self, config):
        if self.connection:
            # Set new configuration
            self.configure(config)

            # Resolve text column
            select = self.resolve(self.text)

            # Initialize reindex operation
            name = self.reindexstart()

            # Copy data over
            self.cursor.execute(Statement.COPY_SECTIONS % (name, select))

            # Stream new results
            self.cursor.execute(Statement.STREAM_SECTIONS % name)
            for uid, text, data, obj, tags in self.rows():
                if not text and self.encoder and obj:
                    yield (uid, self.encoder.decode(obj), tags)
                else:
                    # Read JSON data, if provided
                    data = json.loads(data) if data and isinstance(data, str) else data

                    # Stream data if available, otherwise use section text
                    yield (uid, data if data else text, tags)

            # Swap as new table
            self.cursor.execute(Statement.DROP_SECTIONS)
            self.cursor.execute(Statement.RENAME_SECTIONS % name)

            # Finish reindex operation
            self.reindexend(name)

    def save(self, path):
        if self.connection:
            self.connection.commit()

    def close(self):
        # Close connection
        if self.connection:
            self.connection.close()

    def ids(self, ids):
        # Batch ids and run query
        self.batch(ids=ids)
        self.cursor.execute(Statement.SELECT_IDS)

        # Format and return results
        return self.cursor.fetchall()

    def count(self):
        self.cursor.execute(Statement.COUNT_IDS)
        return self.cursor.fetchone()[0]

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

        # Resolve expression
        if self.expressions and name in self.expressions:
            return self.expressions[name]

        # Name is already resolved, skip
        if name.startswith(self.jsonprefix()) or any(f"s.{s}" == name for s in sections):
            return name

        # Standard columns - need prefixes
        if name.lower() in sections:
            return f"s.{name}"

        # Standard columns - no prefixes
        if name.lower() in noprefix:
            return name

        # Other columns come from documents.data JSON
        return self.jsoncolumn(name)

    def embed(self, similarity, batch):
        # Load similarity results id batch
        self.batch(indexids=[i for i, _ in similarity[batch]], batch=batch)

        # Average and load all similarity scores with first batch
        if not batch:
            self.scores(similarity)

        # Return ids clause placeholder
        return Statement.IDS_CLAUSE % batch

    # pylint: disable=R0912
    def query(self, query, limit, parameters, indexids):
        # Extract query components
        select = query.get("select", self.defaults())
        where = query.get("where")
        groupby, having = query.get("groupby"), query.get("having")
        orderby, qlimit, offset = query.get("orderby"), query.get("limit"), query.get("offset")
        similarity = query.get("similar")

        # Select "indexid, score" when indexids is True
        if indexids:
            select = f"{self.resolve('indexid')}, {self.resolve('score')}"

        # Build query text
        query = Statement.TABLE_CLAUSE % select
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

            # Apply offset
            if offset is not None:
                query += f" OFFSET {offset}"

        # Clear scores when no similar clauses present
        if not similarity:
            self.scores(None)

        # Runs a user query through execute method, which has common user query handling logic
        args = (query, parameters) if parameters else (query,)
        self.execute(self.cursor.execute, *args)

        # Retrieve column list from query
        columns = [c[0] for c in self.cursor.description]

        # Map results and return
        results = []
        for row in self.rows():
            result = {}

            # Copy columns to result. In cases with duplicate column names, find one with a value
            for x, column in enumerate(columns):
                if column not in result or result[column] is None:
                    # Decode object
                    if self.encoder and column == self.object:
                        result[column] = self.encoder.decode(row[x])
                    else:
                        result[column] = row[x]

            results.append(result)

        # Transform results, if necessary
        return [(x["indexid"], x["score"]) for x in results] if indexids else results

    def initialize(self):
        """
        Creates connection and initial database schema if no connection exists.
        """

        if not self.connection:
            # Create database session. Thread locking must be handled externally.
            self.session()

            # Create initial table schema
            self.createtables()

    def session(self, path=None, connection=None):
        """
        Starts a new database session.

        Args:
            path: path to database file
            connection: existing connection to use
        """

        # Create database connection and cursor
        self.connection = connection if connection else self.connect(path) if path else self.connect()
        self.cursor = self.getcursor()

        # Register custom functions - session scope
        self.addfunctions()

        # Create temporary tables - session scope
        self.createbatch()
        self.createscores()

    def createtables(self):
        """
        Creates the initial table schema.
        """

        self.cursor.execute(Statement.CREATE_DOCUMENTS)
        self.cursor.execute(Statement.CREATE_OBJECTS)
        self.cursor.execute(Statement.CREATE_SECTIONS % "sections")
        self.cursor.execute(Statement.CREATE_SECTIONS_INDEX)

    def finalize(self):
        """
        Post processing logic run after inserting a batch of documents. Default method is no-op.
        """

    def loaddocument(self, uid, document, tags, entry):
        """
        Applies pre-processing logic and inserts a document.

        Args:
            uid: unique id
            document: input document dictionary
            tags: document tags
            entry: generated entry date

        Returns:
            section value
        """

        # Make a copy of document before changing
        document = document.copy()

        # Get and remove object field from document
        obj = document.pop(self.object) if self.object in document else None

        # Insert document as JSON
        if document:
            self.insertdocument(uid, json.dumps(document, allow_nan=False), tags, entry)

        # If text and object are both available, load object as it won't otherwise be used
        if self.text in document and obj:
            self.loadobject(uid, obj, tags, entry)

        # Return value to use for section - use text if available otherwise use object
        return document[self.text] if self.text in document else obj

    def insertdocument(self, uid, data, tags, entry):
        """
        Inserts a document.

        Args:
            uid: unique id
            data: document data
            tags: document tags
            entry: generated entry date
        """

        self.cursor.execute(Statement.INSERT_DOCUMENT, [uid, data, tags, entry])

    def loadobject(self, uid, obj, tags, entry):
        """
        Applies pre-preprocessing logic and inserts an object.

        Args:
            uid: unique id
            obj: input object
            tags: object tags
            entry: generated entry date
        """

        # If object support is enabled, save object
        if self.encoder:
            self.insertobject(uid, self.encoder.encode(obj), tags, entry)

    def insertobject(self, uid, data, tags, entry):
        """
        Inserts an object.

        Args:
            uid: unique id
            data: encoded data
            tags: object tags
            entry: generated entry date
        """

        self.cursor.execute(Statement.INSERT_OBJECT, [uid, data, tags, entry])

    def loadsection(self, index, uid, text, tags, entry):
        """
        Applies pre-processing logic and inserts a section.

        Args:
            index: index id
            uid: unique id
            text: section text
            tags: section tags
            entry: generated entry date
        """

        self.insertsection(index, uid, text, tags, entry)

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
        self.cursor.execute(Statement.INSERT_SECTION, [index, uid, text, tags, entry])

    def reindexstart(self):
        """
        Starts a reindex operation.

        Returns:
            temporary working table name
        """

        # Working table name
        name = "rebuild"

        # Create new table to hold reordered sections
        self.cursor.execute(Statement.CREATE_SECTIONS % name)

        return name

    # pylint: disable=W0613
    def reindexend(self, name):
        """
        Ends a reindex operation.

        Args:
            name: working table name
        """

        self.cursor.execute(Statement.CREATE_SECTIONS_INDEX)

    def batch(self, indexids=None, ids=None, batch=None):
        """
        Loads ids to a temporary batch table for efficient query processing.

        Args:
            indexids: list of indexids
            ids: list of ids
            batch: batch index, used when statement has multiple subselects
        """

        # Delete batch when batch id is empty or for batch 0
        if not batch:
            self.cursor.execute(Statement.DELETE_BATCH)

        # Add batch
        self.insertbatch(indexids, ids, batch)

    def createbatch(self):
        """
        Creates temporary batch table.
        """

        # Create or Replace temporary batch table
        self.cursor.execute(Statement.CREATE_BATCH)

    def insertbatch(self, indexids, ids, batch):
        """
        Inserts batch of ids.
        """

        if indexids:
            self.cursor.executemany(Statement.INSERT_BATCH_INDEXID, [(i, batch) for i in indexids])
        if ids:
            self.cursor.executemany(Statement.INSERT_BATCH_ID, [(str(uid), batch) for uid in ids])

    def scores(self, similarity):
        """
        Loads a batch of similarity scores to a temporary table for efficient query processing.

        Args:
            similarity: similarity results as [(indexid, score)]
        """

        # Delete scores
        self.cursor.execute(Statement.DELETE_SCORES)

        if similarity:
            # Average scores per id, needed for multiple similar() clauses
            scores = {}
            for s in similarity:
                for i, score in s:
                    if i not in scores:
                        scores[i] = []
                    scores[i].append(score)

            # Add scores
            self.insertscores(scores)

    def createscores(self):
        """
        Creates temporary scores table.
        """

        # Create or Replace temporary scores table
        self.cursor.execute(Statement.CREATE_SCORES)

    def insertscores(self, scores):
        """
        Inserts a batch of scores.

        Args:
            scores: scores to add
        """

        # Average scores by id
        if scores:
            self.cursor.executemany(Statement.INSERT_SCORE, [(i, sum(s) / len(s)) for i, s in scores.items()])

    def defaults(self):
        """
        Returns a list of default columns when there is no select clause.

        Returns:
            list of default columns
        """

        return "s.id, text, score"

    def connect(self, path=None):
        """
        Creates a new database connection.

        Args:
            path: path to database file

        Returns:
            connection
        """

        raise NotImplementedError

    def getcursor(self):
        """
        Opens a cursor for current connection.

        Returns:
            cursor
        """

        raise NotImplementedError

    def jsonprefix(self):
        """
        Returns json column prefix to test for.

        Returns:
            dynamic column prefix
        """

        raise NotImplementedError

    def jsoncolumn(self, name):
        """
        Builds a json extract column expression for name.

        Args:
            name: column name

        Returns:
            dynamic column expression
        """

        raise NotImplementedError

    def rows(self):
        """
        Returns current cursor row iterator for last executed query.

        Args:
            cursor: cursor

        Returns:
            iterable collection of rows
        """

        raise NotImplementedError

    def addfunctions(self):
        """
        Adds custom functions in current connection.
        """

        raise NotImplementedError
