"""
Task module
"""

import re

import numpy as np
import torch


class Task:
    """
    Base class for all workflow tasks.
    """

    def __init__(self, action=None, select=None, unpack=True, column=None, merge="hstack", initialize=None, finalize=None):
        """
        Creates a new task. A task defines two methods, type of data it accepts and the action to execute
        for each data element. Action is a callable function or list of callable functions.

        Args:
            action: action(s) to execute on each data element
            select: filter(s) used to select data to process
            unpack: if data elements should be unpacked or unwrapped from (id, data, tag) tuples
            column: column index to select if element is a tuple, defaults to all
            merge: merge mode for joining multi-action outputs, defaults to hstack
            initialize: action to execute before processing
            finalize: action to execute after processing
        """

        # Standardize into list of actions
        if not action:
            action = []
        elif not isinstance(action, list):
            action = [action]

        self.action = action
        self.select = select
        self.unpack = unpack
        self.column = column
        self.merge = merge
        self.initialize = initialize
        self.finalize = finalize

    def __call__(self, elements):
        """
        Executes action for a list of data elements.

        Args:
            elements: iterable data elements

        Returns:
            transformed data elements
        """

        if isinstance(elements, list):
            return self.filteredrun(elements)

        return self.run(elements)

    def filteredrun(self, elements):
        """
        Executes a filtered run, which will tag all inputs with a process id, filter elements down to elements the
        task can handle and execute on that subset. Items not selected for processing will be returned unmodified.

        Args:
            elements: iterable data elements

        Returns:
            transformed data elements
        """

        # Build list of elements with unique process ids
        indexed = list(enumerate(elements))

        # Filter data down to data this task handles
        data = [(x, self.upack(element)) for x, element in indexed if self.accept(self.upack(element, True))]

        # Get list of filtered process ids
        ids = [x for x, _ in data]

        # Prepare elements and execute task action(s)
        results = self.execute([self.prepare(element) for _, element in data])

        # Pack results back into elements
        if self.merge:
            elements = self.filteredpack(results, indexed, ids)
        else:
            elements = [self.filteredpack(r, indexed, ids) for r in results]

        return elements

    def filteredpack(self, results, indexed, ids):
        """
        Processes and packs results back into original input elements.

        Args:
            results: task results
            indexed: original elements indexed by process id
            ids: process ids accepted by this task

        Returns:
            packed elements
        """

        # Update with transformed elements. Handle one to many transformations.
        elements = []
        for x, element in indexed:
            if x in ids:
                # Get result for process id
                result = results[ids.index(x)]

                if isinstance(result, OneToMany):
                    # One to many transformations
                    elements.extend([self.pack(element, r) for r in result])
                else:
                    # One to one transformations
                    elements.append(self.pack(element, result))
            else:
                # Pass unprocessed elements through
                elements.append(element)

        return elements

    def run(self, elements):
        """
        Executes a task run for elements. A standard run processes all elements.

        Args:
            elements: iterable data elements

        Returns:
            transformed data elements
        """

        # Execute task actions
        results = self.execute(elements)

        # Handle one to many transformations
        if isinstance(results, list):
            elements = []
            for result in results:
                if isinstance(result, OneToMany):
                    # One to many transformations
                    elements.extend(result)
                else:
                    # One to one transformations
                    elements.append(result)

            return elements

        return results

    def accept(self, element):
        """
        Determines if this task can handle the input data format.

        Args:
            element: input data element

        Returns:
            True if this task can process this data element, False otherwise
        """

        return (isinstance(element, str) and re.search(self.select, element.lower())) if element is not None and self.select else True

    def upack(self, element, force=False):
        """
        Unpacks data for processing.

        Args:
            element: input data element
            force: if true data is unpacked even if task has unpack set to False

        Returns:
            data
        """

        # Extract data from (id, data, tag) formatted elements
        if (self.unpack or force) and isinstance(element, tuple):
            return element[1]

        return element

    def pack(self, element, data):
        """
        Packs data after processing.

        Args:
            element: transformed data element
            data: item to pack element into

        Returns:
            packed data
        """

        # Pack data into (id, data, tag) formatted elements
        if self.unpack and isinstance(element, tuple):
            # If new data is a (id, data, tag) tuple use that except for multi-action "hstack" merges which produce tuples
            if isinstance(data, tuple) and (len(self.action) <= 1 or self.merge != "hstack"):
                return data

            # Create a copy of tuple, update data element and return
            element = list(element)
            element[1] = data
            return tuple(element)

        return data

    def prepare(self, element):
        """
        Method that allows downstream tasks to prepare data element for processing.

        Args:
            element: input data element

        Returns:
            data element ready for processing
        """

        return element

    def execute(self, elements):
        """
        Executes action(s) on elements.

        Args:
            elements: list of data elements

        Returns:
            transformed data elements
        """

        if self.action:
            # Run actions
            outputs = []
            for x, action in enumerate(self.action):
                # Filter elements by column index if necessary - supports a single int or an action index to column index mapping
                index = self.column[x] if isinstance(self.column, dict) else self.column
                inputs = [self.extract(e, index) for e in elements] if index is not None else elements

                # Run action and add outputs
                outputs.append(self.process(action, inputs))

            # Run post process operations
            return self.postprocess(outputs)

        return elements

    def extract(self, element, index):
        """
        Extracts a column from element by index if the element is a tuple.

        Args:
            element: input element
            index: column index

        Returns:
            extracted column
        """

        if isinstance(element, tuple):
            if not self.unpack and len(element) == 3 and isinstance(element[1], tuple):
                return (element[0], element[1][index], element[2])

            return element[index]

        return element

    def process(self, action, inputs):
        """
        Executes action using inputs as arguments.

        Args:
            action: callable object
            inputs: action inputs

        Returns:
            action outputs
        """

        # Execute action and get outputs
        outputs = action(inputs)

        # Wrap one to many transformations if necessary
        if isinstance(outputs, list):
            outputs = [OneToMany(output) if isinstance(output, list) else output for output in outputs]

        return outputs

    def postprocess(self, outputs):
        """
        Runs post process routines after a task action.

        Args:
            outputs: task outputs

        Returns:
            postprocessed outputs
        """

        # Unpack single action tasks
        if len(self.action) == 1:
            return outputs[0]

        # Return unmodified outputs when merge set to None
        if not self.merge:
            return outputs

        if self.merge == "vstack":
            return self.vstack(outputs)
        if self.merge == "concat":
            return self.concat(outputs)

        # Default mode is hstack
        return self.hstack(outputs)

    def vstack(self, outputs):
        """
        Merges outputs row-wise. Returns a list of lists which will be interpreted as a one to many transformation.

        Row-wise merge example (2 actions)

          Inputs: [a, b, c]

          Outputs => [[a1, b1, c1], [a2, b2, c2]]

          Row Merge => [[a1, a2], [b1, b2], [c1, c2]] = [a1, a2, b1, b2, c1, c2]

        Args:
            outputs: task outputs

        Returns:
            list of aggregated/zipped outputs as one to many transforms (row-wise)
        """

        # If all outputs are numpy arrays, use native method
        if all([isinstance(output, np.ndarray) for output in outputs]):
            return np.concatenate(np.stack(outputs, axis=1))

        # If all outputs are torch tensors, use native method
        # pylint: disable=E1101
        if all([torch.is_tensor(output) for output in outputs]):
            return torch.cat(tuple(torch.stack(outputs, axis=1)))

        # Wrap as one to many transforms
        return [OneToMany(x) for x in zip(*outputs)]

    def hstack(self, outputs):
        """
        Merges outputs column-wise. Returns a list of tuples which will be interpreted as a one to one transformation.

        Column-wise merge example (2 actions)

          Inputs: [a, b, c]

          Outputs => [[a1, b1, c1], [a2, b2, c2]]

          Column Merge => [(a1, a2), (b1, b2), (c1, c2)]

        Args:
            outputs: task outputs

        Returns:
            list of aggregated/zipped outputs as tuples (column-wise)
        """

        # If all outputs are numpy arrays, use native method
        if all([isinstance(output, np.ndarray) for output in outputs]):
            return np.stack(outputs, axis=1)

        # If all outputs are torch tensors, use native method
        # pylint: disable=E1101
        if all([torch.is_tensor(output) for output in outputs]):
            return torch.stack(outputs, axis=1)

        return list(zip(*outputs))

    def concat(self, outputs):
        """
        Merges outputs column-wise and concats values together into a string. Returns a list of strings.

        Concat merge example (2 actions)

          Inputs: [a, b, c]

          Outputs => [[a1, b1, c1], [a2, b2, c2]]

          Concat Merge => [(a1, a2), (b1, b2), (c1, c2)] => ["a1. a2", "b1. b2", "c1. c2"]

        Args:
            outputs: task outputs

        Returns:
            list of concat outputs
        """

        return [". ".join([str(y) for y in x if y]) for x in self.hstack(outputs)]


class OneToMany:
    """
    Encapsulates list output for a one to many transformation.
    """

    def __init__(self, values):
        """
        Creates a new OneToMany transformation.

        Args:
            values: list of outputs
        """

        self.values = values

    def __iter__(self):
        return self.values.__iter__()

    def __str__(self):
        return self.values.__str__()
