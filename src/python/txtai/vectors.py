"""
Vectors module
"""

import os
import pickle
import tempfile

from errno import ENOENT
from multiprocessing import Pool

import fasttext
import numpy as np

from pymagnitude import converter, Magnitude
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

from .tokenizer import Tokenizer

# Multiprocessing helper methods
# pylint: disable=W0603
VECTORS = None

def create(path, scoring, blocking):
    """
    Multiprocessing helper method. Creates a global embeddings object to be accessed in a new
    subprocess.

    Args:
        config: configuration
        scoring: scoring instance
    """

    global VECTORS

    # Create a global embedding object using configuration and saved
    VECTORS = WordVectors(path, blocking, scoring)

def transform(document):
    """
    Multiprocessing helper method. Transforms document tokens into an embedding.

    Args:
        document: (id, text|tokens, tags)

    Returns:
        (id, embedding)
    """

    global VECTORS

    return (document[0], VECTORS.transform(document))

class Vectors(object):
    """
    Base class for sentence embeddings/vector models.
    """

    @staticmethod
    def create(method, path, blocking, scoring):
        """
        Create a Vectors model instance.

        Args:
            path: path to word vector model
            blocking: True if method should wait until vectors fully loaded before returning, false otherwise
        """

        # Create vector model instance
        return TransformersVectors(path) if method == "transformers" else WordVectors(path, blocking, scoring)

    def load(self, path, blocking):
        """
        Loads vector model at path.

        Args:
            path: path to word vector model
            blocking: True if method should wait until vectors fully loaded before returning, false otherwise

        Returns:
            vector model
        """

    def index(self, documents):
        """
        Converts a list of document tokens to a temporary file with embeddings arrays. Documents are tuples of (id, text|tokens, tags).
        Returns a tuple of document ids, number of dimensions and temporary file with embeddings.

        Args:
            documents: list of documents

        Returns:
            (ids, dimensions, stream)
        """

    def transform(self, document):
        """
        Transforms document into an embeddings vector.

        Args:
            document: (id, text|tokens, tags)

        Returns:
            embeddings vector
        """

class WordVectors(Vectors):
    """
    Builds sentence embeddings/vectors using weighted word embeddings.
    """

    def __init__(self, path, blocking, scoring):
        self.model = self.load(path, blocking)

        # Store parameters
        self.path = path
        self.blocking = blocking
        self.scoring = scoring

    def load(self, path, blocking):
        # Require that vector path exists, if a path is provided and it's not found, Magnitude will try download from it's servers
        if not path or not os.path.isfile(path):
            raise IOError(ENOENT, "Vector model file not found", path)

        # Load magnitude model. If this is a training run (no embeddings yet), block until the vectors are fully loaded
        return Magnitude(path, case_insensitive=True, blocking=blocking)

    def index(self, documents):
        ids, dimensions, stream = [], None, None

        # Shared objects with Pool
        args = (self.path, self.scoring, self.blocking)

        # Convert all documents to embedding arrays, stream embeddings to disk to control memory usage
        with Pool(os.cpu_count(), initializer=create, initargs=args) as pool:
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy", delete=False) as output:
                stream = output.name
                for uid, embedding in pool.imap(transform, documents):
                    if not dimensions:
                        # Set number of dimensions for embeddings
                        dimensions = embedding.shape[0]

                    ids.append(uid)
                    pickle.dump(embedding, output)

        return (ids, dimensions, stream)

    def transform(self, document):
        # Convert to tokens if necessary
        if isinstance(document[1], str):
            document = (document[0], Tokenizer.tokenize(document[1]), document[2])

        # Generate weights for each vector using a scoring method
        weights = self.scoring.weights(document) if self.scoring else None

        # pylint: disable=E1133
        if weights and [x for x in weights if x > 0]:
            # Build weighted average embeddings vector. Create weights array os float32 to match embeddings precision.
            embedding = np.average(self.lookup(document[1]), weights=np.array(weights, dtype=np.float32), axis=0)
        else:
            # If no weights, use mean
            embedding = np.mean(self.lookup(document[1]), axis=0)

        return embedding

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
        print("Building %d dimension model" % size)

        # Output vectors in vec/txt format
        with open(path + ".txt", "w") as output:
            words = model.get_words()
            output.write("%d %d\n" % (len(words), model.get_dimension()))

            for word in words:
                # Skip end of line token
                if word != "</s>":
                    vector = model.get_word_vector(word)
                    data = ""
                    for v in vector:
                        data += " " + str(v)

                    output.write(word + data + "\n")

        # Build magnitude vectors database
        print("Converting vectors to magnitude format")
        converter.convert(path + ".txt", path + ".magnitude", subword=True)

class TransformersVectors(Vectors):
    """
    Builds sentence embeddings/vectors using the transformers library.
    """

    def __init__(self, path):
        # Sentence transformer model
        self.model = self.load(path, True)

        # Store parameters
        self.path = path

    def load(self, path, blocking):
        model = Transformer(path)
        pooling = Pooling(model.get_word_embedding_dimension())

        return SentenceTransformer(modules=[model, pooling])

    def index(self, documents):
        ids, dimensions, stream = [], None, None

        # Convert all documents to embedding arrays, stream embeddings to disk to control memory usage
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy", delete=False) as output:
            stream = output.name
            batch = []
            for document in documents:
                batch.append(document)

                if len(batch) == 500:
                    # Convert batch to embeddings
                    uids, dimensions = self.batch(batch, output)
                    ids.extend(uids)

                    batch = []

            # Final batch
            if batch:
                uids, dimensions = self.batch(batch, output)
                ids.extend(uids)

        return (ids, dimensions, stream)

    def transform(self, document):
        # Convert to tokens if necessary
        if isinstance(document[1], str):
            document = (document[0], Tokenizer.tokenize(document[1]), document[2])

        return self.model.encode([" ".join(document[1])], show_progress_bar=False)[0]

    def batch(self, documents, output):
        """
        Builds a batch of embeddings.

        Args:
            documents: list of documents used to build embeddings
            output: output temp file to store embeddings

        Returns:
            (ids, dimensions) list of ids and number of dimensions in embeddings
        """

        # Convert to tokens if necessary
        if isinstance(documents[0][1], str):
            documents = [(d[0], Tokenizer.tokenize(d[1]), d[2]) for d in documents]

        # Get list of document texts
        ids = [uid for uid, _, _ in documents]
        documents = [" ".join(tokens) for _, tokens, _ in documents]
        dimensions = None

        # Convert to embeddings
        embeddings = self.model.encode(documents, show_progress_bar=False)
        for embedding in embeddings:
            if not dimensions:
                # Set number of dimensions for embeddings
                dimensions = embedding.shape[0]

            pickle.dump(embedding, output)

        return (ids, dimensions)
