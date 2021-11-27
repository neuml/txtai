"""
StorageTask module
"""

import os
import re

# Conditional import
try:
    from libcloud.storage.providers import get_driver

    LIBCLOUD = True
except ImportError:
    LIBCLOUD = False

from .base import Task


class StorageTask(Task):
    """
    Task that processes object storage buckets. Supports local and cloud providers in Apache libcloud.
    """

    # URL prefix
    PREFIX = r"(\w+):\/\/.*"
    PATH = r"\w+:\/\/(.*)"

    def __init__(self, action=None, select=None, unpack=True):
        if not LIBCLOUD:
            raise ImportError('StorageTask is not available - install "workflow" extra to enable')

        super().__init__(action, select, unpack)

    def __call__(self, elements):
        # Create aggregated directory listing for all elements
        outputs = []
        for element in elements:
            if self.matches(element):
                # Get directory listing and run actions
                outputs.extend(super().__call__(self.list(element)))
            else:
                outputs.append(element)

        return outputs

    def matches(self, element):
        """
        Determines if this element is a storage element.

        Args:
            element: input storage element

        Returns:
            True if this is a storage element
        """

        # Only accept file URLs
        return re.match(StorageTask.PREFIX, self.upack(element, True).lower())

    def list(self, element):
        """
        Gets a list of urls for a object container.

        Args:
            element: object container

        Returns:
            list of urls
        """

        provider = re.sub(StorageTask.PREFIX, r"\1", element.lower())
        path = re.sub(StorageTask.PATH, r"\1", element)

        key, container = os.path.dirname(path), os.path.basename(path)

        client = get_driver(provider)
        driver = client(key)

        container = driver.get_container(container_name=container)
        return [driver.get_object_cdn_url(obj) for obj in driver.list_container_objects(container=container)]
