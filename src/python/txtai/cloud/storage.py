"""
Object storage module
"""

import os

# Conditional import
try:
    from libcloud.storage.providers import get_driver, DRIVERS
    from libcloud.storage.types import ContainerDoesNotExistError, ObjectDoesNotExistError

    LIBCLOUD = True
except ImportError:
    LIBCLOUD, DRIVERS = False, None


from .base import Cloud


class ObjectStorage(Cloud):
    """
    Object storage cloud provider backed by Apache libcloud.
    """

    @staticmethod
    def isprovider(provider):
        """
        Checks if this provider is an object storage provider.

        Args:
            provider: provider name

        Returns:
            True if this is an object storage provider
        """

        return LIBCLOUD and provider and provider.lower() in [x.lower() for x in DRIVERS]

    def __init__(self, config):
        super().__init__(config)

        if not LIBCLOUD:
            raise ImportError('Cloud object storage is not available - install "cloud" extra to enable')

        # Get driver for provider
        driver = get_driver(config["provider"])

        # Get client connection
        self.client = driver(
            config.get("key", os.environ.get("ACCESS_KEY")),
            config.get("secret", os.environ.get("ACCESS_SECRET")),
            **{field: config.get(field) for field in ["host", "port", "region", "token"] if config.get(field)},
        )

    def metadata(self, path=None):
        try:
            # If this is an archive path, check if file exists
            if self.isarchive(path):
                return self.client.get_object(self.config["container"], self.objectname(path))

            # Otherwise check if container exists
            return self.client.get_container(self.config["container"])
        except (ContainerDoesNotExistError, ObjectDoesNotExistError):
            return None

    def load(self, path=None):
        # Download archive file
        if self.isarchive(path):
            obj = self.client.get_object(self.config["container"], self.objectname(path))

            # Create local directory, if necessary
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            obj.download(path, overwrite_existing=True)

        # Download files in container. Optionally filter with a provided prefix.
        else:
            container = self.client.get_container(self.config["container"])
            for obj in container.list_objects(prefix=self.config.get("prefix")):
                # Derive local path and directory
                localpath = os.path.join(path, obj.name)
                directory = os.path.dirname(localpath)

                # Create local directory, if necessary
                os.makedirs(directory, exist_ok=True)

                # Download file locally
                obj.download(localpath, overwrite_existing=True)

        return path

    def save(self, path):
        # Get or create container
        try:
            container = self.client.get_container(self.config["container"])
        except ContainerDoesNotExistError:
            container = self.client.create_container(self.config["container"])

        # Upload files
        for f in self.listfiles(path):
            with open(f, "rb") as iterator:
                self.client.upload_object_via_stream(iterator=iterator, container=container, object_name=self.objectname(f))

    def objectname(self, name):
        """
        Derives an object name. This method checks if a prefix configuration parameter is present and combines
        it with the input name parameter.

        Args:
            name: input name

        Returns:
            object name
        """

        # Get base name
        name = os.path.basename(name)

        # Get optional prefix/folder
        prefix = self.config.get("prefix")

        # Prepend prefix, if applicable
        return f"{prefix}/{name}" if prefix else name
