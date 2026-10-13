"""
SQL module
"""

from io import StringIO
from shlex import shlex

from .error import SQLError
from .expression import Expression


class SQL:
    """
    Translates txtai SQL statements into database native queries.
    """

    # List of clauses to parse
    CLAUSES = ["select", "from", "where", "group", "having", "order", "limit", "offset"]

    def __init__(self, database=None, tolist=False):
        """
        Creates a new SQL query parser.

        Args:
            database: database instance that provides resolver callback, if any
            tolist: outputs expression lists if True, expression text otherwise, defaults to False
        """

        # Expression parser
        self.expression = Expression(database.resolve if database else self.defaultresolve, tolist)

    def __call__(self, query):
        """
        Parses an input SQL query and normalizes column names in the query clauses. This method will also embed
        similarity search placeholders into the query.

        Args:
            query: input query

        Returns:
            {clause name: clause text}
        """

        clauses = None
        if self.issql(query):
            # Multiple statements are not supported
            components = self.statements(query)
            if len(components) > 1:
                raise SQLError("Multiple SQL statements are not supported. Pass values that contain a semicolon as a bind parameter.")
            query = components[0]

            # Tokenize query
            tokens, positions = self.tokenize(query)

            # Alias clauses and similar queries
            aliases, similar = {}, []

            # Parse SQL clauses
            clauses = {
                "select": self.parse(tokens, positions, "select", alias=True, aliases=aliases),
                "where": self.parse(tokens, positions, "where", aliases=aliases, similar=similar),
                "groupby": self.parse(tokens, positions, "group", offset=2, aliases=aliases),
                "having": self.parse(tokens, positions, "having", aliases=aliases),
                "orderby": self.parse(tokens, positions, "order", offset=2, aliases=aliases),
                "limit": self.parse(tokens, positions, "limit", aliases=aliases),
                "offset": self.parse(tokens, positions, "offset", aliases=aliases),
            }

            # Add parsed similar queries, if any
            if similar:
                clauses["similar"] = similar

            # Add select column aliases, if any. Only a query with an offset uses them.
            if aliases and clauses["offset"]:
                clauses["aliases"] = aliases

        # Return clauses, default to full query if this is not a SQL query
        return clauses if clauses else {"similar": [[query]]}

    # pylint: disable=W0613
    def defaultresolve(self, name, alias=None):
        """
        Default resolve function. Performs no processing, only returns name.

        Args:
            name: query column name
            alias: alias name, defaults to None

        Returns:
            name
        """

        return name

    def issql(self, query):
        """
        Detects if this is a SQL query.

        Args:
            query: input query

        Returns:
            True if this is a valid SQL query, False otherwise
        """

        if isinstance(query, str):
            # Reduce query to a lower-cased single line stripped of leading/trailing whitespace
            query = query.lower().strip(";").replace("\n", " ").replace("\t", " ").strip()

            # Detect if this is a valid txtai SQL statement
            return query.startswith("select ") and (" from txtai " in query or query.endswith(" from txtai"))

        return False

    def statements(self, query):
        """
        Splits a SQL query on the semicolons that separate statements.

        A semicolon inside a quoted string is part of the value, not a second statement, so
        the query is scanned rather than split. A doubled quote is an escaped quote and does
        not close the literal.

        Args:
            query: input query

        Returns:
            list of non-empty statements
        """

        components, start, quote = [], 0, None
        index = 0
        while index < len(query):
            char = query[index]
            if quote:
                if char == quote:
                    # A doubled quote is an escaped quote, not the end of the literal
                    if index + 1 < len(query) and query[index + 1] == quote:
                        index += 1
                    else:
                        quote = None
            elif char in ("'", '"'):
                quote = char
            elif char == ";":
                components.append(query[start:index])
                start = index + 1
            index += 1

        components.append(query[start:])

        return [x for x in components if x.strip()]

    def snippet(self, text):
        """
        Parses a partial SQL snippet.

        Args:
            text: SQL snippet

        Returns:
            parsed snippet
        """

        tokens, _ = self.tokenize(text)
        return self.expression(tokens)

    def tokenize(self, query):
        """
        Tokenizes SQL query into tokens.

        Args:
            query: input query

        Returns:
            (tokenized query, token positions)
        """

        # Build a simple SQL lexer
        #   - Punctuation chars are parsed as standalone tokens which helps identify operators
        #   - Add additional wordchars to prevent splitting on those values
        #   - Disable comments
        tokens = shlex(StringIO(query), punctuation_chars="=!<>+-*/%|")
        tokens.wordchars += ":@#"
        tokens.commenters = ""
        tokens = list(tokens)

        # Identify sql clause token positions
        positions = {}

        # Get position of clause keywords. For multi-term clauses, validate next token matches as well
        for x, token in enumerate(tokens):
            t = token.lower()
            if t not in positions and t in SQL.CLAUSES and (t not in ["group", "order"] or (x + 1 < len(tokens) and tokens[x + 1].lower() == "by")):
                positions[t] = x

        return (tokens, positions)

    def parse(self, tokens, positions, name, offset=1, alias=False, aliases=None, similar=None):
        """
        Runs query column name to database column name mappings for clauses. This method will also
        parse SIMILAR() function calls, extract parameters for those calls and leave a placeholder
        to be filled in with similarity results.

        Args:
            tokens: query tokens
            positions: token positions - used to locate the start of sql clauses
            name: current query clause name
            offset: how many tokens are in the clause name
            alias: True if terms in the clause should be aliased (i.e. column as alias)
            aliases: dict of generated aliases, if present these tokens should NOT be resolved
            similar: list where parsed similar clauses should be stored

        Returns:
            formatted clause
        """

        clause = None
        if name in positions:
            # Find the next clause token
            end = [positions.get(x, len(tokens)) for x in SQL.CLAUSES[SQL.CLAUSES.index(name) + 1 :]]
            end = min(end) if end else len(tokens)

            # Start after current clause token and end before next clause or end of string
            clause = tokens[positions[name] + offset : end]

            # Parse and resolve parameters
            clause = self.expression(clause, alias, aliases, similar)

        return clause
