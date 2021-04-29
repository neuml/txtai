"""
Task module
"""

import re


class Task:
    """
    Base class for all workflow tasks.
    """

    def __init__(self, action=None, select=None, unpack=True):
        """
        Creates a new task. A task defines two methods, type of data it accepts and the action to execute
        for each data element. Actions are callable functions.

        Args:
            action: action to execute on each data element
            select: filter(s) used to select data to process
            unpack: if data elements should be unpacked or unwrapped from (id, data, tag) tuples
        """

        self.action = action
        self.select = select
        self.unpack = unpack

    def __call__(self, elements):
        """
        Executes action for a list of data elements.

        Args:
            elements: list of data elements

        Returns:
            transformed data elements
        """

        # Run data preparation routines
        elements = [self.prepare(element) for element in elements]

        # Run data elements through workflow execution
        return self.execute(elements)

    def accept(self, element):
        """
        Determines if this task can handle the input data format.

        Args:
            element: input data element

        Returns:
            True if this task can process this data element, False otherwise
        """

        return (isinstance(element, str) and re.search(self.select, element.lower())) if element and self.select else True

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
        Executes action on elements.

        Args:
            elements: list of data elements

        Returns:
            transformed data elements
        """

        return self.action(elements) if self.action else elements
