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
            elements: iterable data elements

        Returns:
            transformed data elements
        """

        # Run task initializers
        self.initialize()

        # Process elements in batches.
        batch = []
        for x in elements:
            batch.append(x)

            if len(batch) == self.batch:
                yield from self.process(batch)
                batch = []

        # Final batch
        if batch:
            yield from self.process(batch)

        # Run task finalizers
        self.finalize()

    def initialize(self):
        """
        Runs task initializer methods (if any) before processing data.
        """

        # Run task initializers
        for task in self.tasks:
            if task.initialize:
                task.initialize()

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
            data = [(x, task.upack(element)) for x, element in indexed if task.accept(task.upack(element, True))]

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
                        elements.extend([task.pack(element, r) for r in result])
                    else:
                        # One-one transformations
                        elements.append(task.pack(element, result))
                else:
                    # Pass unprocessed elements through
                    elements.append(element)

        # Remove process index and yield results
        yield from elements

    def finalize(self):
        """
        Runs task finalizer methods (if any) after all data processed.
        """

        # Run task finalizers
        for task in self.tasks:
            if task.finalize:
                task.finalize()
