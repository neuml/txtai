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

    def accept(self, element):
        # Only accept file URLs
        return re.match(StorageTask.PREFIX, element.lower())

    def faccept(self, element):
        """
        Determines if this task can handle the input data format.

        Args:
            element: input file url

        Returns:
            True if url is accepted, False otherwise
            True if this task can process this data element, False otherwise
        """

        return super().accept(element)

    def execute(self, elements):
        # Create aggregated directory listing for all elements
        outputs = []
        for element in elements:
            outputs.append(super().execute([url for url in self.list(element) if self.faccept(url)]))

        return outputs

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
