"""
Tabular module
"""

import os

# Conditional import
try:
    import pandas as pd

    PANDAS = True
except ImportError:
    PANDAS = False

from ..base import Pipeline


class Tabular(Pipeline):
    """
    Splits tabular data into rows and columns.
    """

    def __init__(self, idcolumn=None, textcolumns=None, content=False):
        """
        Creates a new Tabular pipeline.

        Args:
            idcolumn: column name to use for row id
            textcolumns: list of columns to combine as a text field
            content: if True, a dict per row is generated with all fields. If content is a list, a subset of fields
                     is included in the generated rows.
        """

        if not PANDAS:
            raise ImportError('Tabular pipeline is not available - install "pipeline" extra to enable')

        self.idcolumn = idcolumn
        self.textcolumns = textcolumns
        self.content = content

    def __call__(self, data):
        """
        Splits data into rows and columns.

        Args:
            data: input data

        Returns:
            list of (id, text, tag)
        """

        items = [data] if not isinstance(data, list) else data

        # Combine all rows into single return element
        results = []
        dicts = []

        for item in items:
            # File path
            if isinstance(item, str):
                _, extension = os.path.splitext(item)
                extension = extension.replace(".", "").lower()

                if extension == "csv":
                    df = pd.read_csv(item)

                results.append(self.process(df))

            # Dict
            if isinstance(item, dict):
                dicts.append(item)

            # List of dicts
            elif isinstance(item, list):
                df = pd.DataFrame(item)
                results.append(self.process(df))

        if dicts:
            df = pd.DataFrame(dicts)
            results.extend(self.process(df))

        return results[0] if not isinstance(data, list) else results

    def process(self, df):
        """
        Extracts a list of (id, text, tag) tuples from a dataframe.

        Args:
            df: DataFrame to extract content from

        Returns:
            list of (id, text, tag)
        """

        rows = []

        # Columns to use for text
        columns = self.textcolumns
        if not columns:
            columns = list(df.columns)
            if self.idcolumn:
                columns.remove(self.idcolumn)

        # Transform into (id, text, tag) tuples
        for index, row in df.iterrows():
            uid = row[self.idcolumn] if self.idcolumn else index
            uid = uid if uid is not None else index
            text = self.concat(row, columns)

            rows.append((uid, text, None))

            # Also add row for content
            if isinstance(self.content, list):
                row = {column: self.column(value) for column, value in row.to_dict().items() if column in self.content}
                rows.append((uid, row, None))
            elif self.content:
                row = {column: self.column(value) for column, value in row.to_dict().items()}
                rows.append((uid, row, None))

        return rows

    def concat(self, row, columns):
        """
        Builds a text field from row using columns.

        Args:
            row: input row
            columns: list of columns to join together

        Returns:
            text
        """

        parts = []
        for column in columns:
            column = self.column(row[column])
            if column:
                parts.append(str(column))

        return ". ".join(parts) if parts else None

    def column(self, value):
        """
        Applies column standardization logic:
            - Replace NaN values with None

        Args:
            value: input value

        Returns:
            formatted value
        """

        return None if pd.isnull(value) else value
