"""
IndexIds module
"""

from ...serialize import Serializer


class IndexIds:
    """
    Stores index ids when content is disabled.
    """

    def __init__(self, embeddings, ids=None):
        """
        Creates an IndexIds instance.

        Args:
            embeddings: embeddings instance
            ids: ids to store
        """

        self.config = embeddings.config
        self.ids = ids

    def __iter__(self):
        yield from self.ids

    def __getitem__(self, index):
        return self.ids[index]

    def __setitem__(self, index, value):
        self.ids[index] = value

    def __add__(self, ids):
        return self.ids + ids

    def load(self, path):
        """
        Loads IndexIds from path.

        Args:
            path: path to load
        """

        if "ids" in self.config:
            # Legacy ids format
            self.ids = self.config.pop("ids")
        else:
            # Standard ids format
            self.ids = Serializer.load(path)

    def save(self, path):
        """
        Saves IndexIds to path.

        Args:
            path: path to save
        """

        Serializer.save(self.ids, path)
