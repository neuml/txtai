"""
Database module
"""

import logging
import types

from inspect import signature

from .encoder import EncoderFactory
from .sql import SQL, SQLError, Token

# Logging configuration
logger = logging.getLogger(__name__)


class Database:
    """
    Base class for database instances. This class encapsulates a document-oriented database
    used for storing key-value content stored as dicts.
    """

    def __init__(self, config):
        """
        Creates a new Database.

        Args:
            config: database configuration
        """

        # Database configuration
        self.config = config

        # SQL parser
        self.sql = SQL(self)

        # Load objects encoder
        encoder = self.config.get("objects")
        self.encoder = EncoderFactory.create(encoder) if encoder else None

        # Load custom functions
        self.functions = self.register(self.config)

    def load(self, path):
        """
        Loads a database path.

        Args:
            path: database url
        """

        raise NotImplementedError

    def insert(self, documents, index=0):
        """
        Inserts documents into the database.

        Args:
            documents: list of documents to save
            index: indexid offset, used for internal ids
        """

        raise NotImplementedError

    def delete(self, ids):
        """
        Deletes documents from database.

        Args:
            ids: ids to delete
        """

        raise NotImplementedError

    def reindex(self, columns=None):
        """
        Reindexes internal database content and streams results back. This method must renumber indexids
        sequentially as deletes could have caused indexid gaps.

        Args:
            columns: optional list of document columns used to rebuild data
        """

        raise NotImplementedError

    def save(self, path):
        """
        Saves a database at path.

        Args:
            path: path to write database
        """

        raise NotImplementedError

    def close(self):
        """
        Closes the database.
        """

        raise NotImplementedError

    def ids(self, ids):
        """
        Retrieves the internal indexids for a list of ids. Multiple indexids may be present for an id in cases
        where data is segmented.

        Args:
            ids: list of document ids

        Returns:
            list of (indexid, id)
        """

        raise NotImplementedError

    def search(self, query, similarity=None, limit=None):
        """
        Runs a search against the database. Supports the following methods:

           1. Standard similarity query. This mode retrieves content for the ids in the similarity results
           2. Similarity query as SQL. This mode will combine similarity results and database results into
              a single result set. Similarity queries are set via the SIMILAR() function.
           3. SQL with no similarity query. This mode runs a SQL query and retrieves the results without similarity queries.

        Example queries:
            "natural language processing" - standard similarity only query
            "select * from txtai where similar('natural language processing')" - similarity query as SQL
            "select * from txtai where similar('nlp') and entry > '2021-01-01'" - similarity query with additional SQL clauses
            "select id, text, score from txtai where similar('nlp')" - similarity query with additional SQL column selections
            "select * from txtai where entry > '2021-01-01' - database only query

        Args:
            query: input query
            similarity: similarity results as [(indexid, score)]
            limit: maximum number of results to return

        Returns:
            query results as a list of dicts
        """

        # Parse query if necessary
        if isinstance(query, str):
            query = self.parse(query)

        # Add in similar results
        where = query.get("where")

        if "select" in query and similarity:
            for x in range(len(similarity)):
                token = f"{Token.SIMILAR_TOKEN}{x}"
                if where and token in where:
                    where = where.replace(token, self.embed(similarity, x))
        elif similarity:
            # Not a SQL query, load similarity results, if any
            where = self.embed(similarity, 0)

        # Save where
        query["where"] = where

        # Run query
        return self.query(query, limit)

    def parse(self, query):
        """
        Parses a query into query components.

        Args:
            query: input query

        Returns:
            dict of parsed query components
        """

        return self.sql(query)

    def resolve(self, name, alias=None):
        """
        Resolves a query column name with the database column name. This method also builds alias expressions
        if alias is set.

        Args:
            name: query column name
            alias: alias name, defaults to None

        Returns:
            database column name
        """

        raise NotImplementedError

    def embed(self, similarity, batch):
        """
        Embeds similarity query results into a database query.

        Args:
            similarity: similarity results as [(indexid, score)]
            batch: batch id
        """

        raise NotImplementedError

    def query(self, query, limit):
        """
        Executes query against database.

        Args:
            query: input query
            limit: maximum number of results to return

        Returns:
            query results
        """

        raise NotImplementedError

    def register(self, config):
        """
        Register custom functions. This method stores the function details for underlying
        database implementations to handle.

        Args:
            config: database configuration
        """

        inputs = config.get("functions") if config else None
        if inputs:
            functions = []
            for fn in inputs:
                name, argcount = None, None

                # Optional function configuration
                if isinstance(fn, dict):
                    name, argcount, fn = fn.get("name"), fn.get("argcount"), fn["function"]

                # Determine if this is a callable object or a function
                if not isinstance(fn, types.FunctionType) and hasattr(fn, "__call__"):
                    name = name if name else fn.__class__.__name__.lower()
                    argcount = argcount if argcount else len(signature(fn.__call__).parameters)
                    fn = fn.__call__
                else:
                    name = name if name else fn.__name__.lower()
                    argcount = argcount if argcount else len(signature(fn).parameters)

                # Store function details
                functions.append((name, argcount, fn))

            return functions

        return None

    def execute(self, function, *args):
        """
        Executes a user query. This method has common error handling logic.

        Args:
            function: database execute function
            args: function arguments

        Returns:
            result of function(args)
        """

        try:
            # Debug log SQL
            logger.debug(*args)

            return function(*args)
        except Exception as ex:
            raise SQLError(ex) from None
