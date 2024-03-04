"""
Vectors module
"""

import pickle
import tempfile

import numpy as np

from ..version import __pickle__


class Vectors:
    """
    Base class for sentence embeddings/vector models. Vector models transform input content into numeric vectors.
    """

    def __init__(self, config, scoring, models):
        """
        Creates a new vectors instance.

        Args:
            config: vector configuration
            scoring: optional scoring instance for term weighting
            models: models cache
        """

        # Store parameters
        self.config = config
        self.scoring = scoring
        self.models = models

        if config:
            # Detect if this is an initialized configuration
            self.initialized = "dimensions" in config

            # Enables optional string tokenization
            self.tokenize = config.get("tokenize")

            # Load model
            self.model = self.load(config.get("path"))

            # Encode batch size - controls underlying model batch size when encoding vectors
            self.encodebatch = config.get("encodebatch", 32)

            # Embeddings instructions
            self.instructions = config.get("instructions")

            # Truncate embeddings to this dimensionality
            self.dimensionality = config.get("dimensionality")

            # Scalar quantization - supports 1-bit through 8-bit quantization
            quantize = config.get("quantize")
            self.qbits = max(min(quantize, 8), 1) if isinstance(quantize, int) and not isinstance(quantize, bool) else None

    def loadmodel(self, path):
        """
        Loads vector model at path.

        Args:
            path: path to vector model

        Returns:
            vector model
        """

        raise NotImplementedError

    def encode(self, data):
        """
        Encodes a batch of data using vector model.

        Args:
            data: batch of data

        Return:
            transformed data
        """

        raise NotImplementedError

    def load(self, path):
        """
        Loads a model using the current configuration. This method will return previously cached models
        if available.

        Returns:
            model
        """

        # Check if model is cached
        if self.models and path in self.models:
            return self.models[path]

        # Create new model
        model = self.loadmodel(path)

        # Store model in cache
        if self.models is not None and path:
            self.models[path] = model

        return model

    def index(self, documents, batchsize=500):
        """
        Converts a list of documents to a temporary file with embeddings arrays. Returns a tuple of document ids,
        number of dimensions and temporary file with embeddings.

        Args:
            documents: list of (id, data, tags)
            batchsize: index batch size

        Returns:
            (ids, dimensions, stream)
        """

        ids, dimensions, batches, stream = [], None, 0, None

        # Convert all documents to embedding arrays, stream embeddings to disk to control memory usage
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy", delete=False) as output:
            stream = output.name
            batch = []
            for document in documents:
                batch.append(document)

                if len(batch) == batchsize:
                    # Convert batch to embeddings
                    uids, dimensions = self.batch(batch, output)
                    ids.extend(uids)
                    batches += 1

                    batch = []

            # Final batch
            if batch:
                uids, dimensions = self.batch(batch, output)
                ids.extend(uids)
                batches += 1

        return (ids, dimensions, batches, stream)

    def transform(self, document):
        """
        Transforms document into an embeddings vector.

        Args:
            document: (id, data, tags)

        Returns:
            embeddings vector
        """

        # Prepare input document for transformers model and build embeddings
        return self.batchtransform([document])[0]

    def batchtransform(self, documents, category=None):
        """
        Transforms batch of documents into embeddings vectors.

        Args:
            documents: list of documents used to build embeddings
            category: category for instruction-based embeddings

        Returns:
            embeddings vectors
        """

        # Prepare input documents for transformers model
        documents = [self.prepare(data, category) for _, data, _ in documents]

        # Skip encoding data if it's already an array
        if documents and isinstance(documents[0], np.ndarray):
            return np.array(documents, dtype=np.float32)

        return self.vectorize(documents)

    def batch(self, documents, output):
        """
        Builds a batch of embeddings.

        Args:
            documents: list of documents used to build embeddings
            output: output temp file to store embeddings

        Returns:
            (ids, dimensions) list of ids and number of dimensions in embeddings
        """

        # Extract ids and prepare input documents for transformers model
        ids = [uid for uid, _, _ in documents]
        documents = [self.prepare(data, "data") for _, data, _ in documents]
        dimensions = None

        # Build embeddings
        embeddings = self.vectorize(documents)
        if embeddings is not None:
            dimensions = embeddings.shape[1]
            pickle.dump(embeddings, output, protocol=__pickle__)

        return (ids, dimensions)

    def prepare(self, data, category=None):
        """
        Prepares input data for vector model.

        Args:
            data: input data
            category: category for instruction-based embeddings

        Returns:
            data formatted for vector model
        """

        # Default instruction category
        category = category if category else "query"

        # Prepend instructions, if applicable
        if self.instructions and category in self.instructions and isinstance(data, str):
            # Prepend category instruction
            data = f"{self.instructions[category]}{data}"

        return data

    def vectorize(self, data):
        """
        Runs data vectorization, which consists of the following steps.

          1. Encode data into vectors using underlying model
          2. Truncate vectors, if necessary
          3. Normalize vectors
          4. Quantize vectors, if necessary

        Args:
            data: input data

        Returns:
            embeddings vectors
        """

        # Transform data into vectors
        embeddings = self.encode(data)

        if embeddings is not None:
            # Truncate embeddings, if necessary
            if self.dimensionality and self.dimensionality < embeddings.shape[1]:
                embeddings = self.truncate(embeddings)

            # Normalize data
            self.normalize(embeddings)

            # Apply quantization, if necessary
            if self.qbits:
                embeddings = self.quantize(embeddings)

        return embeddings

    def truncate(self, embeddings):
        """
        Truncates embeddings to the configured dimensionality.

        This is only useful for models trained to store more important information in
        earlier dimensions such as Matryoshka Representation Learning (MRL).

        Args:
            embeddings: input embeddings

        Returns:
            truncated embeddings
        """

        return embeddings[:, : self.dimensionality]

    def normalize(self, embeddings):
        """
        Normalizes embeddings using L2 normalization. Operation applied directly on array.

        Args:
            embeddings: input embeddings
        """

        # Calculation is different for matrices vs vectors
        if len(embeddings.shape) > 1:
            embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        else:
            embeddings /= np.linalg.norm(embeddings)

    def quantize(self, embeddings):
        """
        Quantizes embeddings using scalar quantization.

        Args:
            embeddings: input embeddings

        Returns:
            quantized embeddings
        """

        # Scale factor is midpoint in range
        factor = 2 ** (self.qbits - 1)

        # Quantize to uint8
        scalars = embeddings * factor
        scalars = scalars.clip(-factor, factor - 1) + factor
        scalars = scalars.astype(np.uint8)

        # Transform uint8 to bits
        bits = np.unpackbits(scalars.reshape(-1, 1), axis=1)

        # Remove unused bits (i.e. for 3-bit quantization, the leading 5 bits are removed)
        bits = bits[:, -self.qbits :]

        # Reshape using original data dimensions and pack bits into uint8 array
        return np.packbits(bits.reshape(embeddings.shape[0], embeddings.shape[1] * self.qbits), axis=1)
