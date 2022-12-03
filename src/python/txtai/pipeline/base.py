"""
Pipeline module
"""


class Pipeline:
    """
    Base class for all Pipelines. The only interface requirement is to define a __call___ method.
    """

    def batch(self, data, size):
        """
        Splits data into separate batch sizes specified by size.

        Args:
            data: data elements
            size: batch size

        Returns:
            list of evenly sized batches with the last batch having the remaining elements
        """

        return [data[x : x + size] for x in range(0, len(data), size)]
