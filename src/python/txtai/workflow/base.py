"""
Workflow module
"""

from .execute import Execute


class Workflow:
    """
    Base class for all workflows.
    """

    def __init__(self, tasks, batch=100, workers=None):
        """
        Creates a new workflow. Workflows are lists of tasks to execute.

        Args:
            tasks: list of workflow tasks
            batch: how many items to process at a time, defaults to 100
            workers: number of concurrent workers
        """

        self.tasks = tasks
        self.batch = batch
        self.workers = workers

        # Set default number of executor workers to max number of actions in a task
        self.workers = max(len(task.action) for task in self.tasks) if not self.workers else self.workers

    def __call__(self, elements):
        """
        Executes a workflow for input elements.

        Args:
            elements: iterable data elements

        Returns:
            transformed data elements
        """

        # Create execute instance for this run
        with Execute(self.workers) as executor:
            # Run task initializers
            self.initialize()

            # Process elements in batches
            for batch in self.chunk(elements):
                yield from self.process(batch, executor)

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

    def chunk(self, elements):
        """
        Splits elements into batches. This method efficiently processes both fixed size inputs and
        dynamically generated inputs.

        Args:
            elements: iterable data elements

        Returns:
            evenly sized batches with the last batch having the remaining elements
        """

        # Build batches by slicing elements, more efficient for fixed sized inputs
        if hasattr(elements, "__len__") and hasattr(elements, "__getitem__"):
            for x in range(0, len(elements), self.batch):
                yield elements[x : x + self.batch]

        # Build batches by iterating over elements when inputs are dynamically generated (i.e. generators)
        else:
            batch = []
            for x in elements:
                batch.append(x)

                if len(batch) == self.batch:
                    yield batch
                    batch = []

            # Final batch
            if batch:
                yield batch

    def process(self, elements, executor):
        """
        Processes a batch of data elements.

        Args:
            elements: iterable data elements
            executor: execute instance, enables concurrent task actions

        Returns:
            transformed data elements
        """

        # Run elements through each task
        for task in self.tasks:
            elements = task(elements, executor)

        # Yield results processed by all tasks
        yield from elements

    def finalize(self):
        """
        Runs task finalizer methods (if any) after all data processed.
        """

        # Run task finalizers
        for task in self.tasks:
            if task.finalize:
                task.finalize()
