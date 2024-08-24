"""
Pickle module
"""

import os
import logging
import pickle
import warnings

from .base import Serialize

# Logging configuration
logger = logging.getLogger(__name__)


class Pickle(Serialize):
    """
    Pickle serialization.
    """

    def __init__(self, allowpickle=False):
        """
        Creates a new instance for Pickle serialization.

        This class ensures the allowpickle parameter or the `ALLOW_PICKLE` environment variable is True. All methods will
        raise errors if this isn't the case.

        Pickle serialization is OK for local data but it isn't recommended when sharing data externally.

        Args:
            allowpickle: default pickle allow mode, only True with methods that generate local temporary data
        """

        # Parent constructor
        super().__init__()

        # Default allow pickle mode
        self.allowpickle = allowpickle

        # Current pickle protocol
        self.version = 4

    def load(self, path):
        # Load pickled data from path, if allowed
        return super().load(path) if self.allow(path) else None

    def save(self, data, path):
        # Save pickled data to path, if allowed
        if self.allow():
            super().save(data, path)

    def loadstream(self, stream):
        # Load pickled data from stream, if allowed
        return pickle.load(stream) if self.allow() else None

    def savestream(self, data, stream):
        # Save pickled data to stream, if allowed
        if self.allow():
            pickle.dump(data, stream, protocol=self.version)

    def loadbytes(self, data):
        # Load pickled data from bytes, if allowed
        return pickle.loads(data) if self.allow() else None

    def savebytes(self, data):
        # Save pickled data to stream, if allowed
        return pickle.dumps(data, protocol=self.version) if self.allow() else None

    def allow(self, path=None):
        """
        Checks if loading and saving pickled data is allowed. Raises an error if it's not allowed.

        Args:
            path: optional path to add to generated error messages
        """

        enablepickle = self.allowpickle or os.environ.get("ALLOW_PICKLE", "True") in ("True", "1")
        if not enablepickle:
            raise ValueError(
                (
                    "Loading of pickled index data is disabled. "
                    f"`{path if path else 'stream'}` was not loaded. "
                    "Set the env variable `ALLOW_PICKLE=True` to enable loading pickled index data. "
                    "This should only be done for trusted and/or local data."
                )
            )

        if not self.allowpickle:
            warnings.warn(
                (
                    "Pickled index data formats are deprecated and loading will be disabled by default in the future. "
                    "Set the env variable `ALLOW_PICKLE=False` to disable the loading of pickled index data formats. "
                    "Saving this index will replace pickled index data formats with the latest index formats and remove this warning."
                ),
                FutureWarning,
            )

        return enablepickle
