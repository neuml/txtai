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

    def __call__(self, elements, executor=None):
        # Run task
        outputs = super().__call__(elements, executor)

        # Get output file extension
        output = self.output
        parts = list(os.path.splitext(output))
        extension = parts[-1].lower()

        # Add timestamp to filename
        if self.timestamp:
            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            parts[-1] = timestamp + parts[-1]

            # Create full path to output file
            output = ".".join(parts)

        # Write output
        if extension == ".xlsx":
            pd.DataFrame(outputs).to_excel(output, index=False)
        else:
            pd.DataFrame(outputs).to_csv(output, index=False)

        # Return results
        return outputs
