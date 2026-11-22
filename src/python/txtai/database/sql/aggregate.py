"""
Aggregate module
"""

import itertools
import operator

from .base import SQL
from .error import SQLError
from .token import Token


class Aggregate(SQL):
    """
    Aggregates partial results from queries. Partial results come from queries when working with sharded indexes.
    """

    def __init__(self, database=None):
        # Always return token lists as this method requires them
        super().__init__(database, True)

    def __call__(self, query, results):
        """
        Analyzes query results, combines aggregate function results and applies ordering.

        Args:
            query: input query
            results: query results

        Returns:
            aggregated query results
        """

        # Parse query
        query = super().__call__(query)

        # Check if this is a SQL query with results. A sharded query that matches
        # nothing across all shards yields empty results, so guard before indexing.
        if "select" in query and results:
            # Get list of unique and aggregate columns. If no aggregate columns or order by found, skip
            columns = list(results[0].keys())
            expressions = self.expressions(query, columns)
            aggcolumns = self.aggcolumns(columns, expressions)
            if aggcolumns or query["orderby"]:
                # Merge aggregate columns
                if aggcolumns:
                    results = self.aggregate(query, results, columns, expressions, aggcolumns)

                # Sort results and return
                return self.orderby(query, results, columns, expressions) if query["orderby"] else self.defaultsort(results)

        # Otherwise, run default sort
        return self.defaultsort(results)

    def expressions(self, query, columns):
        """
        Maps each result column to the select expression it holds. Aliased columns are returned under the alias name,
        so they are mapped to the expression before the alias. All other columns map to themselves.

        Args:
            query: parsed query
            columns: list of result columns

        Returns:
            {column: expression}
        """

        aliases = {}
        for alias, index in query.get("aliases", {}).items():
            # Remove the alias name and the optional alias keyword from the select expression
            tokens, _ = self.tokenize(query["select"][index])
            tokens = tokens[:-1]
            if tokens and Token.isaliastoken(tokens[-1]):
                tokens = tokens[:-1]

            aliases[alias] = self.expression.buildtext(tokens)

        return {column: aliases.get(Token.normalize(column), column) for column in columns}

    def aggcolumns(self, columns, expressions):
        """
        Filters columns for columns that have an aggregate function call.

        Args:
            columns: list of columns
            expressions: {column: expression}

        Returns:
            list of aggregate columns
        """

        aggregates = {}
        for column in columns:
            # Match function names case-insensitively but keep the column name as returned
            name = expressions[column].lower()
            if name.startswith(("count(", "sum(", "total(")):
                aggregates[column] = sum
            elif name.startswith("max("):
                aggregates[column] = max
            elif name.startswith("min("):
                aggregates[column] = min
            elif name.startswith("avg("):
                aggregates[column] = self.avg

        return aggregates

    def avg(self, column, values, counts):
        """
        Combines per-shard/per-group average values into a single average, weighted by a
        matching count(col) column when one was selected. count(col) only counts the rows where
        col is not NULL, which are the rows avg(col) is computed over. Without it, there is no way
        to recombine the averages correctly, so this raises instead of returning a wrong number.

        Args:
            column: avg column name
            values: list of per-shard/per-group average values
            counts: list of matching count(col) values, or None

        Returns:
            combined average
        """

        if len(values) == 1:
            return values[0]

        if not counts or sum(counts) == 0:
            raise SQLError(f"{column} requires a count({column[4:]} column to combine results from multiple shards")

        return sum(value * count for value, count in zip(values, counts)) / sum(counts)

    def aggregate(self, query, results, columns, expressions, aggcolumns):
        """
        Merges aggregate columns in results.

        Args:
            query: input query
            results: query results
            columns: list of select columns
            expressions: {column: expression}
            aggcolumns: list of aggregate columns

        Returns:
            results with aggregates merged
        """

        # Group data, if necessary
        if query["groupby"]:
            results = self.groupby(query, results, columns, expressions)
        else:
            results = [results]

        # Map each avg(col) column to its count(col) column, the non-NULL row count used to weight it
        countnames = {expressions[column][6:]: column for column in columns if expressions[column].lower().startswith("count(")}
        countcolumns = {column: countnames.get(expressions[column][4:]) for column in aggcolumns if expressions[column].lower().startswith("avg(")}

        # Compute column values
        rows = []
        for result in results:
            # Calculate/copy column values
            row = {}
            for column in columns:
                if column in aggcolumns:
                    # Calculate aggregate value, skip NULLs from shards with no matching rows
                    matches = [r for r in result if r[column] is not None]
                    countcolumn = countcolumns.get(column)
                    values, counts = [r[column] for r in matches], [r[countcolumn] for r in matches] if countcolumn else None
                    if not values:
                        row[column] = None
                    elif column in countcolumns:
                        row[column] = self.avg(expressions[column], values, counts)
                    else:
                        row[column] = aggcolumns[column](values)
                else:
                    # Non aggregate column value repeat, use first value
                    row[column] = result[0][column]

            # Add row using original query columns
            rows.append(row)

        return rows

    def groupby(self, query, results, columns, expressions):
        """
        Groups results using query group by clause.

        Args:
            query: input query
            results: query results
            columns: list of select columns
            expressions: {column: expression}

        Returns:
            results grouped using group by clause
        """

        # Group by columns are matched case-insensitively, either by column name or by the expression an alias names
        names = {name.lower() for name in query["groupby"]}
        groupby = [column for column in columns if column.lower() in names or expressions[column].lower() in names]
        if groupby:
            # NULLs sort first like in SQLite
            results = sorted(results, key=lambda row: [(row[column] is not None, row[column]) for column in groupby])
            return [list(value) for _, value in itertools.groupby(results, operator.itemgetter(*groupby))]

        return [results]

    def orderby(self, query, results, columns, expressions):
        """
        Applies an order by clause to results.

        Args:
            query: input query
            results: query results
            columns: list of select columns
            expressions: {column: expression}

        Returns:
            results ordered using order by clause
        """

        # Sort in reverse order
        for clause in query["orderby"][::-1]:
            # Order by columns must be selected
            reverse = False
            if clause.lower().endswith(" asc"):
                clause = clause.rsplit(" ", 1)[0].strip()
            elif clause.lower().endswith(" desc"):
                clause = clause.rsplit(" ", 1)[0].strip()
                reverse = True

            # Match order by column case-insensitively, either by column name or by the expression an alias names
            clause = clause.lower()
            column = next((column for column in columns if clause in (column.lower(), expressions[column].lower())), None)

            # Order by columns must be in select clause. NULLs sort first like in SQLite (last when descending).
            if column:
                results = sorted(results, key=lambda row, column=column: (row[column] is not None, row[column]), reverse=reverse)

        return results

    def defaultsort(self, results):
        """
        Default sorting algorithm for results. Sorts by score descending, if available.

        Args:
            results: query results

        Returns:
            results ordered by score descending
        """

        # Sort standard query using score column, if present
        if results and "score" in results[0]:
            return sorted(results, key=lambda x: (x["score"] is not None, x["score"]), reverse=True)

        return results
