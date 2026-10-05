"""
TurboVec module
"""

# Conditional import
try:
    from turbovec import IdMapIndex

    TURBOVEC = True
except ImportError:
    TURBOVEC = False

from ..base import ANN

# Core library imports
from ...util import Library

np = Library().numpy()


class TurboVec(ANN):
    """
    Builds an ANN index using the turbovec library.
    """

    def __init__(self, config):
        super().__init__(config)

        if not TURBOVEC:
            raise ImportError('turbovec is not available - install "ann" extra to enable')

    def load(self, path):
        # Load index
        self.backend = IdMapIndex.load(path)

    def index(self, embeddings):
        # Lookup index settings
        bitwidth = self.setting("bitwidth", 4)

        # Create index
        self.backend = IdMapIndex(dim=self.config["dimensions"], bit_width=bitwidth)

        # Add items - position in embeddings is used as the id
        self.backend.add_with_ids(embeddings, np.arange(embeddings.shape[0], dtype=np.uint64))

        # Add id offset, delete counter and index build metadata
        self.config["offset"] = embeddings.shape[0]
        self.metadata({"bitwidth": bitwidth})

    def append(self, embeddings):
        new = embeddings.shape[0]

        self.backend.add_with_ids(embeddings, np.arange(self.config["offset"], self.config["offset"] + new, dtype=np.uint64))

        # Update id offset and index metadata
        self.config["offset"] += new
        self.metadata()

    def delete(self, ids):
        # Mark elements as deleted to omit from search results
        for uid in ids:
            self.backend.remove(uid)

    def search(self, queries, limit):
        # Run the query
        scores, ids = self.backend.search(queries=queries, k=limit)

        # Map results to [(id, score)]
        results = []
        for x, score in enumerate(scores):

            # Build (id, score) tuples, convert to ints and floats
            results.append(list(zip(ids[x].tolist(), score.tolist())))

        return results

    def count(self):
        return len(self.backend)

    def save(self, path):
        # Write index
        self.backend.write(path)
