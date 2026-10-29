"""
ExportTask module
"""

import datetime
import os

# Conditional import
try:
    import pandas as pd

    PANDAS = True
except ImportError:
    PANDAS = False

from .base import Task


class ExportTask(Task):
    """
    Task that exports task elements using Pandas.
    """

    def register(self, output=None, timestamp=None):
        """
        Add export parameters to task. Checks if required dependencies are installed.

        Args:
            output: output file path
            timestamp: true if output file should be timestamped
        """

        if not PANDAS:
            raise ImportError('ExportTask is not available - install "workflow" extra to enable')

        # pylint: disable=W0201
        self.output = output
        self.timestamp = timestamp

        # Rows exported and output file path for the current run
        self.rows, self.path = [], None

        # Workflows run this task once per batch. Start a new export when a workflow run starts.
        self.initializer, self.initialize = self.initialize, self.start

    def __call__(self, elements, executor=None):
        # Run task
        outputs = super().__call__(elements, executor)

        # Add outputs to the rows exported for the current run
        self.rows.extend(outputs)

        # Get output file extension
        output = self.output
        parts = list(os.path.splitext(output))
        extension = parts[-1].lower()

        # Resolve output file path once per run
        # pylint: disable=W0201
        if not self.path:
            # Add timestamp to filename
            if self.timestamp:
                timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                parts[-1] = timestamp + parts[-1]

                # Create full path to output file
                output = ".".join(parts)

            self.path = output

        # Write all rows exported for the current run
        if extension == ".xlsx":
            pd.DataFrame(self.rows).to_excel(self.path, index=False)
        else:
            pd.DataFrame(self.rows).to_csv(self.path, index=False)

        # Return results
        return outputs

    def start(self):
        """
        Starts a new export and runs the initialize action, if any.
        """

        # pylint: disable=W0201
        self.rows, self.path = [], None

        if self.initializer:
            self.initializer()
