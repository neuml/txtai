"""
Ids module
"""


class Ids:
    """
    Resolves internal ids for lists of ids.
    """

    def __init__(self, embeddings):
        """
        Create a new ids action.

        Args:
            embeddings: embeddings instance
        """

        self.database = embeddings.database
        self.ids = embeddings.ids

    def __call__(self, ids):
        """
        Resolve internal ids.

        Args:
            ids: ids

        Returns:
            internal ids
        """

        # Resolve ids using database if available, otherwise fallback to embeddings ids
        results = self.database.ids(ids) if self.database else self.scan(ids)

        # Create dict of id: [iids] given there is a one to many relationship
        ids = {}
        for iid, uid in results:
            if uid not in ids:
                ids[uid] = []
            ids[uid].append(iid)

        return ids

    def scan(self, ids):
        """
        Scans embeddings ids array for matches when content is disabled.

        Args:
            ids: search ids

        Returns:
            internal ids
        """

        # Find existing ids
        indices = []
        for uid in ids:
            indices.extend([(index, value) for index, value in enumerate(self.ids) if uid == value])

        return indices
