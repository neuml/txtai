"""
Word Vectors module
"""

import logging
import os
import pickle
import tempfile

from errno import ENOENT
from multiprocessing import Pool

import numpy as np

# Conditionally import Word Vector libraries as they aren't installed by default
try:
    import fasttext
    from pymagnitude import converter, Magnitude

    WORDS = True
except ImportError:
    WORDS = False

from .. import __pickle__

from .base import Vectors
from ..pipeline import Tokenizer

# Logging configuration
logger = logging.getLogger(__name__)

# Multiprocessing helper methods
# pylint: disable=W0603
VECTORS = None


def create(config, scoring):
    """
    Multiprocessing helper method. Creates a global embeddings object to be accessed in a new subprocess.

    Args:
        config: vector configuration
        scoring: scoring instance
    """

    global VECTORS

    # Create a global embedding object using configuration and saved
    VECTORS = WordVectors(config, scoring)


def transform(document):
    """
    Multiprocessing helper method. Transforms document into an embeddings vector.

    Args:
        document: (id, data, tags)

    Returns:
        (id, embedding)
    """

    return (document[0], VECTORS.transform(document))


class WordVectors(Vectors):
    """
    Builds sentence embeddings/vectors using weighted word embeddings.
    """

    def load(self, path):
        # Ensure that vector path exists
        if not path or not os.path.isfile(path):
            raise IOError(ENOENT, "Vector model file not found", path)

        # Load magnitude model. If this is a training run (uninitialized config), block until vectors are fully loaded
        return Magnitude(path, case_insensitive=True, blocking=not self.initialized)

    def encode(self, data):
        # Iterate over each data element, tokenize (if necessary) and build an aggregated embeddings vector
        embeddings = []
        for tokens in data:
            # Convert to tokens if necessary
            if isinstance(tokens, str):
                tokens = Tokenizer.tokenize(tokens)

            # Generate weights for each vector using a scoring method
            weights = self.scoring.weights(tokens) if self.scoring else None

            # pylint: disable=E1133
            if weights and [x for x in weights if x > 0]:
                # Build weighted average embeddings vector. Create weights array os float32 to match embeddings precision.
                embedding = np.average(self.lookup(tokens), weights=np.array(weights, dtype=np.float32), axis=0)
            else:
                # If no weights, use mean
                embedding = np.mean(self.lookup(tokens), axis=0)

            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    def index(self, documents, batchsize=1):
        # Use default single process indexing logic
        if "parallel" in self.config and not self.config["parallel"]:
            return super().index(documents, batchsize)

        # Customize indexing logic with multiprocessing pool to efficiently build vectors
        ids, dimensions, batches, stream = [], None, 0, None

        # Shared objects with Pool
        args = (self.config, self.scoring)

        # Convert all documents to embedding arrays, stream embeddings to disk to control memory usage
        with Pool(os.cpu_count(), initializer=create, initargs=args) as pool:
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy", delete=False) as output:
                stream = output.name
                embeddings = []
                for uid, embedding in pool.imap(transform, documents):
                    if not dimensions:
                        # Set number of dimensions for embeddings
                        dimensions = embedding.shape[0]

                    ids.append(uid)
                    embeddings.append(embedding)

                    if len(embeddings) == batchsize:
                        pickle.dump(np.array(embeddings, dtype=np.float32), output, protocol=__pickle__)
                        batches += 1

                        embeddings = []

                # Final embeddings batch
                if embeddings:
                    pickle.dump(np.array(embeddings, dtype=np.float32), output, protocol=__pickle__)
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

        return self.model.query(tokens)

    @staticmethod
    def isdatabase(path):
        """
        Checks if this is a SQLite database file which is the file format used for word vectors databases.

        Args:
            path: path to check

        Returns:
            True if this is a SQLite database
        """

        if isinstance(path, str) and os.path.isfile(path) and os.path.getsize(path) >= 100:
            # Read 100 byte SQLite header
            with open(path, "rb") as f:
                header = f.read(100)

            # Check for SQLite header
            return header.startswith(b"SQLite format 3\000")

        return False

    @staticmethod
    def build(data, size, mincount, path):
        """
        Builds fastText vectors from a file.

        Args:
            data: path to input data file
            size: number of vector dimensions
            mincount: minimum number of occurrences required to register a token
            path: path to output file
        """

        # Train on data file using largest dimension size
        model = fasttext.train_unsupervised(data, dim=size, minCount=mincount)

        # Output file path
        logger.info("Building %d dimension model", size)

        # Output vectors in vec/txt format
        with open(path + ".txt", "w", encoding="utf-8") as output:
            words = model.get_words()
            output.write(f"{len(words)} {model.get_dimension()}\n")

            for word in words:
                # Skip end of line token
                if word != "</s>":
                    vector = model.get_word_vector(word)
                    data = ""
                    for v in vector:
                        data += " " + str(v)

                    output.write(word + data + "\n")

        # Build magnitude vectors database
        logger.info("Converting vectors to magnitude format")
        converter.convert(path + ".txt", path + ".magnitude", subword=True)
