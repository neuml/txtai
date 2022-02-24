"""
Cloud module
"""

import os

# Conditional import
try:
    from libcloud.storage.providers import get_driver
    from libcloud.storage.types import ContainerDoesNotExistError, ObjectDoesNotExistError

    LIBCLOUD = True
except ImportError:
    LIBCLOUD = False


class Cloud:
    """
    Methods to sync files with cloud storage.
    """

    def __init__(self, config):
        """
        Creates a new cloud storage client connection.

        Args:
            config: cloud storage configuration
        """

        if not LIBCLOUD:
            raise ImportError('Cloud storage is not available - install "cloud" extra to enable')

        self.config = config

        # Get driver for provider
        driver = get_driver(config["provider"])

        # Get client connection
        self.client = driver(
            config.get("key", os.environ.get("ACCESS_KEY")),
            config.get("secret", os.environ.get("ACCESS_SECRET")),
            host=config.get("host"),
            port=config.get("port"),
            token=config.get("token"),
            region=config.get("region"),
        )

    def exists(self, path):
        """
        Checks if path exists.

        Args:
            path: path to check

        Returns:
            True if path exists, False otherwise
        """

        try:
            self.client.get_object(self.config["container"], os.path.basename(path))
            return True
        except (ContainerDoesNotExistError, ObjectDoesNotExistError):
            return False

    def load(self, path):
        """
        Retrieves file from cloud storage and stores locally.

        Args:
            path: path to retrieve
        """

        # Download object
        obj = self.client.get_object(self.config["container"], os.path.basename(path))
        obj.download(path, overwrite_existing=True)

    def save(self, path):
        """
        Sends local file to cloud storage.

        Args:
            path: path to send
        """

        # Get or create container
        try:
            container = self.client.get_container(self.config["container"])
        except ContainerDoesNotExistError:
            container = self.client.create_container(self.config["container"])

        # Upload object
        with open(path, "rb") as iterator:
            self.client.upload_object_via_stream(iterator=iterator, container=container, object_name=os.path.basename(path))
