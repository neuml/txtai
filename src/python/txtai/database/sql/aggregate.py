"""
Aggregate module
"""

import itertools
import operator

from .base import SQL
from .error import SQLError


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
            aggcolumns = self.aggcolumns(columns)
            if aggcolumns or query["orderby"]:
                # Merge aggregate columns
                if aggcolumns:
                    results = self.aggregate(query, results, columns, aggcolumns)

                # Sort results and return
                return self.orderby(query, results) if query["orderby"] else self.defaultsort(results)

        # Otherwise, run default sort
        return self.defaultsort(results)

    def aggcolumns(self, columns):
        """
        Filters columns for columns that have an aggregate function call.

        Args:
            columns: list of columns

        Returns:
            list of aggregate columns
        """

        aggregates = {}
        for column in columns:
            column = column.lower()
            if column.startswith(("count(", "sum(", "total(")):
                aggregates[column] = sum
            elif column.startswith("max("):
                aggregates[column] = max
            elif column.startswith("min("):
                aggregates[column] = min
            elif column.startswith("avg("):
                aggregates[column] = self.avg

        return aggregates

    def avg(self, values, counts):
        """
        Combines per-shard/per-group average values into a single average, weighted by a
        matching count(*) column when one was selected. Without it, there is no way to
        recombine the averages correctly, so this raises instead of returning a wrong number.

        Args:
            values: list of per-shard/per-group average values
            counts: list of matching count(*) values, or None

        Returns:
            combined average
        """

        if len(values) == 1:
            return values[0]

        if not counts or sum(counts) == 0:
            raise SQLError("avg() requires a count(*) column to combine results from multiple shards")

        return sum(value * count for value, count in zip(values, counts)) / sum(counts)

    def aggregate(self, query, results, columns, aggcolumns):
        """
        Merges aggregate columns in results.

        Args:
            query: input query
            results: query results
            columns: list of select columns
            aggcolumns: list of aggregate columns

        Returns:
            results with aggregates merged
        """

        # Group data, if necessary
        if query["groupby"]:
            results = self.groupby(query, results, columns)
        else:
            results = [results]

        # Column providing the row count each group's values were computed over, used to weight avg() columns
        countcolumn = next((column for column in columns if column.lower() == "count(*)"), None)

        # Compute column values
        rows = []
        for result in results:
            # Row counts for this group, if a count(*) column was selected
            counts = [r[countcolumn] for r in result] if countcolumn else None

            # Calculate/copy column values
            row = {}
            for column in columns:
                if column in aggcolumns:
                    # Calculate aggregate value
                    values = [r[column] for r in result]
                    row[column] = self.avg(values, counts) if column.lower().startswith("avg(") else aggcolumns[column](values)
                else:
                    # Non aggregate column value repeat, use first value
                    row[column] = result[0][column]

            # Add row using original query columns
            rows.append(row)

        return rows

    def groupby(self, query, results, columns):
        """
        Groups results using query group by clause.

        Args:
            query: input query
            results: query results
            columns: list of select columns

        Returns:
            results grouped using group by clause
        """

        groupby = [column for column in columns if column.lower() in query["groupby"]]
        if groupby:
            results = sorted(results, key=operator.itemgetter(*groupby))
            return [list(value) for _, value in itertools.groupby(results, operator.itemgetter(*groupby))]

        return [results]

    def orderby(self, query, results):
        """
        Applies an order by clause to results.

        Args:
            query: input query
            results: query results

        Returns:
            results ordered using order by clause
        """

        # Sort in reverse order
        for clause in query["orderby"][::-1]:
            # Order by columns must be selected
            reverse = False
            if clause.lower().endswith(" asc"):
                clause = clause.rsplit(" ")[0]
            elif clause.lower().endswith(" desc"):
                clause = clause.rsplit(" ")[0]
                reverse = True

            # Order by columns must be in select clause
            if clause in query["select"]:
                results = sorted(results, key=operator.itemgetter(clause), reverse=reverse)

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
            return sorted(results, key=lambda x: x["score"], reverse=True)

        return results
