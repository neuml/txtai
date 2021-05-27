"""
Workflow module
"""


class Workflow:
    """
    Base class for all workflows.
    """

    def __init__(self, tasks, batch=100):
        """
        Creates a new workflow. Workflows are lists of tasks to execute.

        Args:
            tasks: list of workflow tasks
            batch: how many items to process at a time, defaults to 100
        """

        self.tasks = tasks
        self.batch = batch

    def __call__(self, elements):
        """
        Executes a workflow for input elements.

        Args:
            elements: list of data elements

        Returns:
            transformed data elements
        """

        batch = []
        for x in elements:
            batch.append(x)

            if len(batch) == self.batch:
                yield from self.process(batch)
                batch = []

        if batch:
            yield from self.process(batch)

    def process(self, elements):
        """
        Processes a batch of data elements.

        Args:
            elements: list of data elements

        Returns:
            transformed data elements
        """

        for task in self.tasks:
            # Build list of elements with unique process ids
            indexed = list(enumerate(elements))

            # Filter data down to data this task handles
            data = [(x, self.unpack(element) if task.unpack else element) for x, element in indexed if task.accept(self.unpack(element))]

            # Get list of filtered process ids
            ids = [x for x, _ in data]

            # Execute task action
            results = task([element for _, element in data])

            # Update with transformed elements. Handle one to many transformations.
            elements = []
            for x, element in indexed:
                if x in ids:
                    # Get result for process id
                    result = results[ids.index(x)]

                    if isinstance(result, list):
                        # One-many transformations
                        elements.extend([self.pack(element, r) if task.unpack else r for r in result])
                    else:
                        # One-one transformations
                        elements.append(self.pack(element, result) if task.unpack else result)
                else:
                    # Pass unprocessed elements through
                    elements.append(element)

        # Remove process index and yield results
        yield from elements

    def unpack(self, element):
        """
        Unpacks data for processing.

        Args:
            element: input data element

        Returns:
            data
        """

        # Extract data from (id, data, tag) formatted elements
        if isinstance(element, tuple):
            return element[1]

        return element

    def pack(self, element, data):
        """
        Packs data for delivery to the next workflow task.

        Args:
            element: transformed data element
            data: item to pack element into

        Returns:
            packed data
        """

        # Package data into (id, data, tag) formatted elements
        if isinstance(element, tuple):
            # If new data is a (id, data, tag) tuple use that
            if isinstance(data, tuple):
                return data

            # Create a copy of tuple, update data element and return
            element = list(element)
            element[1] = data
            return tuple(element)

        return data
