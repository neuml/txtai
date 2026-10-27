"""
SQL module
"""

import re
from io import StringIO
from shlex import shlex

from .error import SQLError
from .expression import Expression

# Matches a mantissa followed by a bare exponent letter, e.g. "1e", "1.5E".
# shlex splits on the "-"/"+" in "1e-3", so these must be rejoined with the
# following sign and digits to keep the numeric literal intact.
_EXPONENT_PREFIX = re.compile(r"^(\d+\.?\d*|\.\d+)[eE]$")
_UNSIGNED_INT = re.compile(r"^\d+$")


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
            components = [x for x in query.split(";") if x.strip()]
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

        # Rejoin exponent-notation literals that shlex split on the sign:
        # "1e", "-", "3" -> "1e-3"; "1.5E", "+", "10" -> "1.5E+10".
        merged = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if _EXPONENT_PREFIX.match(token) and i + 2 < len(tokens) and tokens[i + 1] in ("+", "-") and _UNSIGNED_INT.match(tokens[i + 2]):
                merged.append(token + tokens[i + 1] + tokens[i + 2])
                i += 3
            else:
                merged.append(token)
                i += 1
        tokens = merged

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
