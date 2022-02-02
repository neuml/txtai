"""
Workflow module
"""

import logging
import time
import traceback

from datetime import datetime

# Conditional import
try:
    from croniter import croniter

    CRONITER = True
except ImportError:
    CRONITER = False

from .execute import Execute

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Workflow:
    """
    Base class for all workflows.
    """

    def __init__(self, tasks, batch=100, workers=None, name=None):
        """
        Creates a new workflow. Workflows are lists of tasks to execute.

        Args:
            tasks: list of workflow tasks
            batch: how many items to process at a time, defaults to 100
            workers: number of concurrent workers
            name: workflow name
        """

        self.tasks = tasks
        self.batch = batch
        self.workers = workers
        self.name = name

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

    def schedule(self, cron, elements, iterations=None):
        """
        Schedules a workflow using a cron expression and elements.

        Args:
            cron: cron expression
            elements: iterable data elements passed to workflow each call
            iterations: number of times to run workflow, defaults to run indefinitely
        """

        # Check that croniter is installed
        if not CRONITER:
            raise ImportError('Workflow scheduling is not available - install "workflow" extra to enable')

        logger.info("'%s' scheduler started with schedule %s", self.name, cron)

        maxiterations = iterations
        while iterations is None or iterations > 0:
            # Schedule using localtime
            schedule = croniter(cron, datetime.now().astimezone()).get_next(datetime)
            logger.info("'%s' next run scheduled for %s", self.name, schedule.isoformat())
            time.sleep(schedule.timestamp() - time.time())

            # Run workflow
            # pylint: disable=W0703
            try:
                for _ in self(elements):
                    pass
            except Exception:
                logger.error(traceback.format_exc())

            # Decrement iterations remaining, if necessary
            if iterations is not None:
                iterations -= 1

        logger.info("'%s' max iterations (%d) reached", self.name, maxiterations)

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
