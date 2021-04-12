"""
Transformers module
"""

import pickle
import tempfile

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

from ..pipeline.tokenizer import Tokenizer

from .base import Vectors
from ..models import Models


class TransformersVectors(Vectors):
    """
    Builds sentence embeddings/vectors using the transformers library.
    """

    def load(self, path):
        modelhub = self.config.get("modelhub", True)

        # Download model from the model hub (default)
        if modelhub:
            model = Transformer(path)
            pooling = Pooling(model.get_word_embedding_dimension())

            # Detect unbounded tokenizer typically found in older models
            Models.checklength(model.auto_model, model.tokenizer)

            return SentenceTransformer(modules=[model, pooling])

        # Download model directly from sentence transformers if model hub disabled
        return SentenceTransformer(path)

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
