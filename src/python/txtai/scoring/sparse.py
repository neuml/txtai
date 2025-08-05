"""
Sparse module
"""

from queue import Queue
from threading import Thread

from ..ann import SparseANNFactory
from ..vectors import SparseVectorsFactory

from .base import Scoring


class Sparse(Scoring):
    """
    Sparse vector scoring.
    """

    # End of stream message
    COMPLETE = 1

    def __init__(self, config=None, models=None):
        super().__init__(config)

        # Vector configuration
        mapping = {"vectormethod": "method", "vectornormalize": "normalize"}
        config = {k: v for k, v in config.items() if k not in mapping.values()}
        for k, v in mapping.items():
            if k in config:
                config[v] = config[k]

        # Load the SparseVectors model
        self.model = SparseVectorsFactory.create(config, models)

        # Normalize search outputs if vectors are not normalized already
        # A float can also be provided to set the normalization factor (defaults to 30.0)
        self.isnormalize = self.config.get("normalize", True)

        # Sparse ANN
        self.ann = None

        # Encoding processing parameters
        self.batch = self.config.get("batch", 1024)
        self.thread, self.queue, self.data = None, None, None

    def insert(self, documents, index=None, checkpoint=None):
        # Start processing thread, if necessary
        self.start(checkpoint)

        data = []
        for uid, document, tags in documents:
            # Extract text, if necessary
            if isinstance(document, dict):
                document = document.get(self.text, document.get(self.object))

            if document is not None:
                # Add data
                data.append((uid, " ".join(document) if isinstance(document, list) else document, tags))

        # Add batch of data
        self.queue.put(data)

    def delete(self, ids):
        self.ann.delete(ids)

    def index(self, documents=None):
        # Insert documents, if provided
        if documents:
            self.insert(documents)

        # Create ANN, if there is pending data
        embeddings = self.stop()
        if embeddings is not None:
            self.ann = SparseANNFactory.create(self.config)
            self.ann.index(embeddings)

    def upsert(self, documents=None):
        # Insert documents, if provided
        if documents:
            self.insert(documents)

        # Check for existing index and pending data
        if self.ann:
            embeddings = self.stop()
            if embeddings is not None:
                self.ann.append(embeddings)
        else:
            self.index()

    def weights(self, tokens):
        # Not supported
        return None

    def search(self, query, limit=3):
        return self.batchsearch([query], limit)[0]

    def batchsearch(self, queries, limit=3, threads=True):
        # Convert queries to embedding vectors
        embeddings = self.model.batchtransform((None, query, None) for query in queries)

        # Run ANN search
        scores = self.ann.search(embeddings, limit)

        # Normalize scores if normalization IS enabled AND vector normalization IS NOT enabled
        return self.normalize(embeddings, scores) if self.isnormalize and not self.model.isnormalize else scores

    def count(self):
        return self.ann.count()

    def load(self, path):
        self.ann = SparseANNFactory.create(self.config)
        self.ann.load(path)

    def save(self, path):
        # Save Sparse ANN
        if self.ann:
            self.ann.save(path)

    def close(self):
        # Close Sparse ANN
        if self.ann:
            self.ann.close()

        # Clear parameters
        self.model, self.ann, self.thread, self.queue = None, None, None, None

    def issparse(self):
        return True

    def isnormalized(self):
        return self.isnormalize or self.model.isnormalize

    def start(self, checkpoint):
        """
        Starts an encoding processing thread.

        Args:
            checkpoint: checkpoint directory
        """

        if not self.thread:
            self.queue = Queue(5)
            self.thread = Thread(target=self.encode, args=(checkpoint,))
            self.thread.start()

    def stop(self):
        """
        Stops an encoding processing thread. Return processed results.

        Returns:
            results
        """

        results = None
        if self.thread:
            # Send EOS message
            self.queue.put(Sparse.COMPLETE)

            self.thread.join()
            self.thread, self.queue = None, None

            # Get return value
            results = self.data
            self.data = None

        return results

    def encode(self, checkpoint):
        """
        Encodes streaming data.

        Args:
            checkpoint: checkpoint directory
        """

        # Streaming encoding of data
        _, dimensions, self.data = self.model.vectors(self.stream(), self.batch, checkpoint)

        # Save number of dimensions
        self.config["dimensions"] = dimensions

    def stream(self):
        """
        Streams data from an input queue until end of stream message received.
        """

        batch = self.queue.get()
        while batch != Sparse.COMPLETE:
            yield from batch
            batch = self.queue.get()

    def normalize(self, queries, scores):
        """
        Normalize query result using the max query score.

        Args:
            queries: query vectors
            scores: query results

        Returns:
            normalized query results
        """

        # Get normalize scale factor
        scale = 30.0 if isinstance(self.isnormalize, bool) else self.isnormalize

        # Normalize scores using max scores
        maxscores = self.model.dot(queries, queries)

        # Normalize results and return
        results = []
        for x, result in enumerate(scores):
            maxscore = max(maxscores[x][x] / scale, scale)
            maxscore = max(maxscore, result[0][1]) if result else maxscore

            results.append([(uid, score / maxscore) for uid, score in result])

        return results
