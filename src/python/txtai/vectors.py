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
    def create(config, scoring):
        """
        Create a Vectors model instance.

        Args:
            config: vector configuration
            scoring: scoring instance
        """

        # Derive vector type
        transformers = config.get("method") == "transformers"

        # Create vector model instance
        return TransformersVectors(config, scoring) if transformers else WordVectors(config, scoring)

    def __init__(self, config, scoring):
        # Store parameters
        self.config = config
        self.scoring = scoring

        # Detect if this is an initialized configuration
        self.initialized = "dimensions" in config

        # Enables optional string tokenization
        self.tokenize = "tokenize" not in config or config["tokenize"]

        # pylint: disable=E1111
        self.model = self.load(config["path"])

    def load(self, path):
        """
        Loads vector model at path.

        Args:
            path: path to word vector model

        Returns:
            vector model
        """

    def index(self, documents):
        """
        Converts a list of documents to a temporary file with embeddings arrays. Returns a tuple of document ids,
        number of dimensions and temporary file with embeddings.

        Args:
            documents: list of (id, text|tokens, tags)

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

    def load(self, path):
        # Ensure that vector path exists
        if not path or not os.path.isfile(path):
            raise IOError(ENOENT, "Vector model file not found", path)

        # Load magnitude model. If this is a training run (uninitialized config), block until vectors are fully loaded
        return Magnitude(path, case_insensitive=True, blocking=not self.initialized)

    def index(self, documents):
        ids, dimensions, stream = [], None, None

        # Shared objects with Pool
        args = (self.config, self.scoring)

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

    def load(self, path):
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
        # Convert input document to text and build embeddings
        return self.model.encode([self.text(document[1])], show_progress_bar=False)[0]

    def batch(self, documents, output):
        """
        Builds a batch of embeddings.

        Args:
            documents: list of documents used to build embeddings
            output: output temp file to store embeddings

        Returns:
            (ids, dimensions) list of ids and number of dimensions in embeddings
        """

        # Extract ids and convert input documents to text
        ids = [uid for uid, _, _ in documents]
        documents = [self.text(text) for _, text, _ in documents]
        dimensions = None

        # Build embeddings
        embeddings = self.model.encode(documents, show_progress_bar=False)
        for embedding in embeddings:
            if not dimensions:
                # Set number of dimensions for embeddings
                dimensions = embedding.shape[0]

            pickle.dump(embedding, output)

        return (ids, dimensions)

    def text(self, text):
        """
        Converts input into text that can be processed by transformer models. This method supports
        optional string tokenization and joins tokenized input into text.

        Args:
            text: text|tokens
        """

        # Optional string tokenization
        if self.tokenize and isinstance(text, str):
            text = Tokenizer.tokenize(text)

        # Transformer models require string input
        if isinstance(text, list):
            text = " ".join(text)

        return text
