"""
Word Vectors module
"""

import json
import logging
import os

from multiprocessing import Pool

# Conditional import
try:
    from staticvectors import Database, StaticVectors

    STATICVECTORS = True
except ImportError:
    STATICVECTORS = False

from ...pipeline import Tokenizer
from ...util import Download, DownloadError, Library

from ..base import Vectors
from ..recovery import Recovery

# Core library imports
np = Library().numpy()

# Logging configuration
logger = logging.getLogger(__name__)

# Multiprocessing helper methods
# pylint: disable=W0603
PARAMETERS, VECTORS = None, None


def create(config, scoring):
    """
    Multiprocessing helper method. Creates a global embeddings object to be accessed in a new subprocess.

    Args:
        config: vector configuration
        scoring: scoring instance
    """

    global PARAMETERS
    global VECTORS

    # Store model parameters for lazy loading
    PARAMETERS, VECTORS = (config, scoring, None), None


def transform(document):
    """
    Multiprocessing helper method. Transforms document into an embeddings vector.

    Args:
        document: (id, data, tags)

    Returns:
        (id, embedding)
    """

    # Lazy load vectors model
    global VECTORS
    if not VECTORS:
        VECTORS = WordVectors(*PARAMETERS)

    return (document[0], VECTORS.transform(document))


class WordVectors(Vectors):
    """
    Builds vectors using weighted word embeddings.
    """

    @staticmethod
    def ismodel(path):
        """
        Checks if path is a WordVectors model.

        Args:
            path: input path

        Returns:
            True if this is a WordVectors model, False otherwise
        """

        # Check if this is a SQLite database
        if WordVectors.isdatabase(path):
            return True

        try:
            # Download file and parse JSON
            path = Download()(path, "config.json")
            if path:
                with open(path, encoding="utf-8") as f:
                    config = json.load(f)
                    return config.get("model_type") == "staticvectors"

        # Ignore this error - invalid repo or directory
        except DownloadError:
            pass

        return False

    @staticmethod
    def isdatabase(path):
        """
        Checks if this is a SQLite database file which is the file format used for word vectors databases.

        Args:
            path: path to check

        Returns:
            True if this is a SQLite database
        """

        return isinstance(path, str) and STATICVECTORS and Database.isdatabase(path)

    def __init__(self, config, scoring, models):
        # Check before parent constructor since it calls loadmodel
        if not STATICVECTORS:
            raise ImportError('staticvectors is not available - install "vectors" extra to enable')

        super().__init__(config, scoring, models)

    def loadmodel(self, path):
        return StaticVectors(path)

    def encode(self, data, category=None):
        # Iterate over each data element, tokenize (if necessary) and build an aggregated embeddings vector
        embeddings = []
        for tokens in data:
            # Convert to tokens, if necessary. If tokenized list is empty, use input string.
            if isinstance(tokens, str):
                tokenlist = Tokenizer.tokenize(tokens)
                tokens = tokenlist if tokenlist else [tokens]

            # Generate weights for each vector using a scoring method
            weights = self.scoring.weights(tokens) if self.scoring else None

            # pylint: disable=E1133
            if weights and [x for x in weights if x > 0]:
                # Build weighted average embeddings vector. Create weights array as float32 to match embeddings precision.
                embedding = np.average(self.lookup(tokens), weights=np.array(weights, dtype=np.float32), axis=0)
            else:
                # If no weights, use mean
                embedding = np.mean(self.lookup(tokens), axis=0)

            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    def index(self, documents, batchsize=500, checkpoint=None):
        # Derive number of parallel processes
        parallel = self.config.get("parallel", True)
        parallel = os.cpu_count() if parallel and isinstance(parallel, bool) else int(parallel)

        # Use default single process indexing logic
        if not parallel:
            return super().index(documents, batchsize, checkpoint)

        # Customize indexing logic with multiprocessing pool to efficiently build vectors
        ids, dimensions, batches, stream = [], None, 0, None

        # Shared objects with Pool
        args = (self.config, self.scoring)

        # Generate recovery config if checkpoint is set
        vectorsid = self.vectorsid() if checkpoint else None
        recovery = Recovery(checkpoint, vectorsid, self.loadembeddings) if checkpoint else None

        # Convert all documents to embedding arrays, stream embeddings to disk to control memory usage
        with Pool(parallel, initializer=create, initargs=args) as pool:
            with self.spool(checkpoint, vectorsid) as output:
                stream = output.name
                batch = []
                for document in documents:
                    batch.append(document)

                    if len(batch) == batchsize:
                        uids, dimensions = self.parallelbatch(pool, batch, output, recovery)
                        ids.extend(uids)
                        batches += 1

                        batch = []

                # Final batch
                if batch:
                    uids, dimensions = self.parallelbatch(pool, batch, output, recovery)
                    ids.extend(uids)
                    batches += 1

        return (ids, dimensions, batches, stream)

    def parallelbatch(self, pool, documents, output, recovery):
        """
        Builds a batch of embeddings using the multiprocessing pool, honoring a recovery
        checkpoint if one is available for this batch.

        Args:
            documents: list of documents used to build embeddings
            pool: multiprocessing pool used to transform documents not served from recovery
            output: output stream to store embeddings
            recovery: optional recovery instance

        Returns:
            (ids, dimensions) list of ids and number of dimensions in embeddings
        """

        ids = [uid for uid, _, _ in documents]

        # Attempt to read embeddings from a recovery file
        embeddings = recovery() if recovery else None
        if embeddings is None:
            embeddings = np.array([embedding for _, embedding in pool.imap(transform, documents, self.encodebatch)], dtype=np.float32)

        dimensions = embeddings.shape[1]
        self.saveembeddings(output, embeddings)

        return (ids, dimensions)

    def lookup(self, tokens):
        """
        Queries word vectors for given list of input tokens.

        Args:
            tokens: list of tokens to query

        Returns:
            word vectors array
        """

        return self.model.embeddings(tokens)

    def tokens(self, data):
        # Skip tokenization rules
        return data
