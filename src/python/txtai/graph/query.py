"""
Query module
"""

import logging
import re
from io import StringIO
from shlex import shlex

try:
    from grandcypher import GrandCypher

    GRANDCYPHER = True
except ImportError:
    GRANDCYPHER = False

# Logging configuration
logger = logging.getLogger(__name__)


class Query:
    """
    Runs openCypher graph queries using the GrandCypher library. This class also supports search functions.
    """

    # Similar token
    SIMILAR = "__SIMILAR__"

    def __init__(self):
        """
        Create a new graph query instance.
        """

        if not GRANDCYPHER:
            raise ImportError('GrandCypher is not available - install "graph" extra to enable')

    def __call__(self, graph, query, limit):
        """
        Runs a graph query.

        Args:
            graph: graph instance
            query: graph query, can be a full query string or a parsed query dictionary
            limit: number of results

        Returns:
            results
        """

        # Results by attribute and ids filter
        attributes, uids = None, None

        # Build the query from a parsed query
        if isinstance(query, dict):
            query, attributes, uids = self.build(query)

        # Filter graph, if applicable
        if uids:
            graph = self.filter(graph, attributes, uids)

        # Debug log graph query
        logger.debug(query)

        # Run openCypher query
        return GrandCypher(graph.backend, limit if limit else 3).run(query)

    def isquery(self, queries):
        """
        Checks a list of queries to see if all queries are openCypher queries.

        Args:
            queries: list of queries to check

        Returns:
            True if all queries are openCypher queries
        """

        # Check for required graph query clauses
        return all(query and query.strip().startswith("MATCH ") and "RETURN " in query for query in queries)

    def parse(self, query):
        """
        Parses a graph query. This method supports parsing search functions and replacing them with placeholders.

        Args:
            query: graph query

        Returns:
            parsed query as a dictionary
        """

        # Parameters
        where, limit, offset, nodes, similar = None, None, None, [], []

        # Parse where clause
        match = re.search(r"where(.+?)return", query, flags=re.DOTALL | re.IGNORECASE)
        if match:
            where = match.group(1).strip()

        # Parse limit clause
        match = re.search(r"limit\s+(\d+)", query, flags=re.DOTALL | re.IGNORECASE)
        if match:
            limit = match.group(1)

        # Parse skip clause when results are ordered first by score desc. Graph results with no ORDER BY do not follow score order.
        # Most queries have no SKIP, so check for the keyword before the regex.
        if "skip" in query.lower():
            match = re.search(r"skip\s+(\d+)", query, flags=re.DOTALL | re.IGNORECASE)
            if match and re.search(r"order\s+by\s+\w+\.score\s+desc", query, flags=re.DOTALL | re.IGNORECASE):
                offset = match.group(1)

        # Parse similar clauses
        clauses, spans = self.clauses(query), []
        for x, (start, end, params) in enumerate(clauses):
            spans.append((start, end, f"{Query.SIMILAR}{x}"))
            nodes.append(params[0])
            similar.append(params[1:])

        # Replace similar clauses with placeholders
        for start, end, placeholder in reversed(spans):
            query = query[:start] + placeholder + query[end:]

        # Return parsed query
        return {
            "query": query,
            "where": where,
            "limit": limit,
            "offset": offset,
            "nodes": nodes,
            "similar": similar,
        }

    def clauses(self, query):
        """
        Finds similar clauses in a graph query. Quoted text is kept as is, including commas and parentheses.

        Args:
            query: graph query

        Returns:
            list of (start, end, parameters) for each similar clause
        """

        if "similar" not in query.lower():
            return []

        # Tokenize query, quotes are kept in quoted tokens
        try:
            tokens = shlex(StringIO(query))
            tokens.commenters = ""
            tokens = list(tokens)
        except ValueError:
            # Unbalanced quotes
            return []

        # Find the character position of each token
        positions, offset = [], 0
        for token in tokens:
            offset = query.index(token, offset)
            positions.append((offset, offset + len(token)))
            offset += len(token)

        clauses, x = [], 0
        while x < len(tokens) - 1:
            if tokens[x].lower() == "similar" and tokens[x + 1] == "(":
                # Split parameters on commas at the top level of the clause
                params, depth, y = [[]], 1, x + 2
                while y < len(tokens):
                    token = tokens[y]
                    depth += (token == "(") - (token == ")")
                    if depth == 0:
                        break

                    if token == "," and depth == 1:
                        params.append([])
                    else:
                        params[-1].append(y)
                    y += 1

                if depth == 0:
                    # Parameter text is unquoted when it's a single quoted token
                    values = []
                    for param in params:
                        value = query[positions[param[0]][0] : positions[param[-1]][1]] if param else ""
                        values.append(value[1:-1] if len(param) == 1 and value[:1] in ("'", '"') else value.replace("'", "").replace('"', ""))

                    clauses.append((positions[x][0], positions[y][1], [value.strip() for value in values]))
                    x = y
            x += 1

        return clauses

    def build(self, parse):
        """
        Constructs a full query from a parsed query. This method supports substituting placeholders with search results.

        Args:
            parse: parsed query

        Returns:
            graph query
        """

        # Get query. Initialize attributes and uids.
        query, attributes, uids = parse["query"], {}, {}

        # Replace similar clause with id query
        if "results" in parse:
            for x, result in enumerate(parse["results"]):
                # Get query node
                node = parse["nodes"][x]

                # Add similar match attribute
                attribute = f"match_{x}"
                clause = f"{node}.{attribute} > 0"

                # Replace placeholder with earch results
                query = query.replace(f"{Query.SIMILAR}{x}", f"{clause}")

                # Add uids and scores
                for uid, score in result:
                    if uid not in uids:
                        uids[uid] = score

                # Add results by attribute matched
                attributes[attribute] = result

        # Return query, results by attribute matched and ids filter
        return query, attributes, uids.items()

    def filter(self, graph, attributes, uids):
        """
        Filters the input graph by uids. This method also adds similar match attributes.

        Args:
            graph: graph instance
            attributes: results by attribute matched
            uids: single list with all matching ids

        Returns:
            filtered graph
        """

        # Filter the graph
        graph = graph.filter(uids)

        # Add similar match attributes
        for attribute, result in attributes.items():
            for uid, score in result:
                graph.addattribute(uid, attribute, score)

        return graph
