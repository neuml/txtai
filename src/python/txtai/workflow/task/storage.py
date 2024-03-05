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

    def register(self, key=None, secret=None, host=None, port=None, token=None, region=None):
        """
        Checks if required dependencies are installed. Reads in cloud storage parameters.

        Args:
            key: provider-specific access key
            secret: provider-specific access secret
            host: server host name
            port: server port
            token: temporary session token
            region: storage region
        """

        if not LIBCLOUD:
            raise ImportError('StorageTask is not available - install "workflow" extra to enable')

        # pylint: disable=W0201
        self.key = key
        self.secret = secret
        self.host = host
        self.port = port
        self.token = token
        self.region = region

    def __call__(self, elements, executor=None):
        # Create aggregated directory listing for all elements
        outputs = []
        for element in elements:
            if self.matches(element):
                # Get directory listing and run actions
                outputs.extend(super().__call__(self.list(element), executor))
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

        # Load key and secret, if applicable
        key = self.key if self.key is not None else os.environ.get("ACCESS_KEY")
        secret = self.secret if self.secret is not None else os.environ.get("ACCESS_SECRET")

        # Parse key and container
        key, container = (os.path.dirname(path), os.path.basename(path)) if key is None else (key, path)

        # Parse optional prefix from container
        parts = container.split("/", 1)
        container, prefix = (parts[0], parts[1]) if len(parts) > 1 else (container, None)

        # Get driver for provider
        driver = get_driver(provider)

        # Get client connection
        client = driver(key, secret, **{field: getattr(self, field) for field in ["host", "port", "region", "token"] if getattr(self, field)})

        container = client.get_container(container_name=container)
        return [client.get_object_cdn_url(obj) for obj in client.list_container_objects(container=container, prefix=prefix)]
