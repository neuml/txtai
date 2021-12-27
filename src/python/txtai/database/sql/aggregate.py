"""
Aggregate module
"""

import itertools
import operator

from .base import SQL


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

        # Check if this is a SQL query
        if "select" in query:
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
                aggregates[column] = lambda x: sum(x) / len(x)

        return aggregates

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

        # Compute column values
        rows = []
        for result in results:
            # Calculate/copy column values
            row = {}
            for column in columns:
                if column in aggcolumns:
                    # Calculate aggregate value
                    function = aggcolumns[column]
                    row[column] = function([r[column] for r in result])
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
