"""
Query module
"""

import logging
import re

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
        where, limit, nodes, similar = None, None, [], []

        # Parse where clause
        match = re.search(r"where(.+?)return", query, flags=re.DOTALL | re.IGNORECASE)
        if match:
            where = match.group(1).strip()

        # Parse limit clause
        match = re.search(r"limit\s+(\d+)", query, flags=re.DOTALL | re.IGNORECASE)
        if match:
            limit = match.group(1)

        # Parse similar clauses
        for x, match in enumerate(re.finditer(r"similar\((.+?)\)", query, flags=re.DOTALL | re.IGNORECASE)):
            # Replace similar clause with placeholder
            query = query.replace(match.group(0), f"{Query.SIMILAR}{x}")

            # Parse similar clause parameters
            params = [param.strip().replace("'", "").replace('"', "") for param in match.group(1).split(",")]
            nodes.append(params[0])
            similar.append(params[1:])

        # Return parsed query
        return {
            "query": query,
            "where": where,
            "limit": limit,
            "nodes": nodes,
            "similar": similar,
        }

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
