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

    def __init__(self, action=None, select=None, unpack=True, ids=True):
        if not LIBCLOUD:
            raise ImportError('StorageTask is not available - install "workflow" extra to enable')

        super().__init__(action, select, unpack)

        # If true, elements returned will be tagged with ids and converted into (id, data, tag) tuples
        self.ids = ids

    def accept(self, element):
        # Only accept file URLs
        return super().accept(element) and re.match(StorageTask.PREFIX, element.lower())

    def execute(self, elements):
        # List contents of each bucket
        buckets = [self.list(element) for element in elements]

        elements = []
        for bucket in buckets:
            # Execute actions
            content = super().execute(bucket)

            # Combine with content ids if necessary
            if self.ids:
                values = []
                for x, url in enumerate(bucket):
                    values.append((url, content[x], None))
            else:
                values = content

            elements.append(values)

        return elements

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
