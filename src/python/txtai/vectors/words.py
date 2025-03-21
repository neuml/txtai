"""
Word Vectors module
"""

import json
import logging
import os
import tempfile

from multiprocessing import Pool

import numpy as np

from huggingface_hub.errors import HFValidationError
from transformers.utils import cached_file

# Conditional import
try:
    from staticvectors import Database, StaticVectors

    STATICVECTORS = True
except ImportError:
    STATICVECTORS = False

from ..pipeline import Tokenizer

from .base import Vectors

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
            path = cached_file(path_or_repo_id=path, filename="config.json")
            if path:
                with open(path, encoding="utf-8") as f:
                    config = json.load(f)
                    return config.get("model_type") == "staticvectors"

        # Ignore this error - invalid repo or directory
        except (HFValidationError, OSError):
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

    def encode(self, data):
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
            return super().index(documents, batchsize)

        # Customize indexing logic with multiprocessing pool to efficiently build vectors
        ids, dimensions, batches, stream = [], None, 0, None

        # Shared objects with Pool
        args = (self.config, self.scoring)

        # Convert all documents to embedding arrays, stream embeddings to disk to control memory usage
        with Pool(parallel, initializer=create, initargs=args) as pool:
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy", delete=False) as output:
                stream = output.name
                embeddings = []
                for uid, embedding in pool.imap(transform, documents, self.encodebatch):
                    if not dimensions:
                        # Set number of dimensions for embeddings
                        dimensions = embedding.shape[0]

                    ids.append(uid)
                    embeddings.append(embedding)

                    if len(embeddings) == batchsize:
                        np.save(output, np.array(embeddings, dtype=np.float32))
                        batches += 1

                        embeddings = []

                # Final embeddings batch
                if embeddings:
                    np.save(output, np.array(embeddings, dtype=np.float32))
                    batches += 1

        return (ids, dimensions, batches, stream)

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
