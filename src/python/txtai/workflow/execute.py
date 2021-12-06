"""
Execute module
"""

from multiprocessing.pool import Pool, ThreadPool

import torch.multiprocessing


class Execute:
    """
    Supports sequential, multithreading and multiprocessing based execution of tasks.
    """

    def __init__(self, workers=None):
        """
        Creates a new execute instance. Functions can be executed sequentially, in a thread pool
        or in a process pool. Once created, the thread and/or process pool will stay open until the
        close method is called.

        Args:
            workers: number of workers for thread/process pools
        """

        # Number of workers to use in thread/process pools
        self.workers = workers

        self.thread = None
        self.process = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        self.close()

    def run(self, method, function, args):
        """
        Runs multiple calls of function for each tuple in args. The method parameter controls if the calls are
        sequential (method = None), multithreaded (method = "thread") or with multiprocessing (method="process").

        Args:
            method: run method - "thread" for multithreading, "process" for multiprocessing, otherwise runs sequentially
            function: function to run
            args: list of tuples with arguments to each call
        """

        # Concurrent processing
        if method and len(args) > 1:
            pool = self.pool(method)
            if pool:
                return pool.starmap(function, args, 1)

        # Sequential processing
        return [function(*arg) for arg in args]

    def pool(self, method):
        """
        Gets a handle to a concurrent processing pool. This method will create the pool if it doesn't already exist.

        Args:
            method: pool type - "thread" or "process"

        Returns:
            concurrent processing pool or None if no pool of that type available
        """

        if method == "thread":
            if not self.thread:
                self.thread = ThreadPool(self.workers)

            return self.thread

        if method == "process":
            if not self.process:
                # Importing torch.multiprocessing will register torch shared memory serialization for cuda
                self.process = Pool(self.workers, context=torch.multiprocessing.get_context("spawn"))

            return self.process

        return None

    def close(self):
        """
        Closes concurrent processing pools.
        """

        if self.thread:
            self.thread.close()
            self.thread.join()
            self.thread = None

        if self.process:
            self.process.close()
            self.process.join()
            self.process = None
