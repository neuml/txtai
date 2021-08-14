"""
Transformers module
"""

import pickle
import tempfile

import numpy as np
import torch

# Conditional import
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS = True
except ImportError:
    SENTENCE_TRANSFORMERS = False

from transformers import AutoModel, AutoTokenizer

from ..pipeline.tokenizer import Tokenizer

from .base import Vectors
from ..models import Models


class TransformersVectors(Vectors):
    """
    Builds sentence embeddings/vectors using the transformers library.
    """

    def load(self, path):
        # Flag that determines if transformers or sentence-transformers should be used to run model
        modelhub = self.config.get("modelhub", True)

        # Tensor device id
        deviceid = Models.deviceid(self.config.get("gpu", True))

        # Use transformers model directly (default)
        if modelhub:
            model = AutoModel.from_pretrained(path)
            tokenizer = AutoTokenizer.from_pretrained(path)

            # Detect unbounded tokenizer typically found in older models
            Models.checklength(model, tokenizer)

            return (model, tokenizer, Models.device(deviceid))

        if not SENTENCE_TRANSFORMERS:
            raise ImportError('sentence-transformers is not available - install "similarity" extra to enable')

        # Download model directly from sentence transformers if model hub disabled
        return SentenceTransformer(path, device=Models.reference(deviceid))

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
        return self.encode([self.text(document[1])])[0]

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
        embeddings = self.encode(documents)
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

    def encode(self, documents):
        """
        Builds embeddings for input documents.

        Args:
            documents: list of text to build embeddings for

        Returns:
            embeddings
        """

        # Build embeddings using transformers
        if isinstance(self.model, tuple):
            # Unpack model
            model, tokenizer, device = self.model

            # Move model to device
            model = model.to(device)

            # Split documents into batches and process
            results = []
            for chunk in self.chunk(documents, 32):
                # Tokenize input
                inputs = tokenizer(chunk, padding=True, truncation="longest_first", return_tensors="pt")

                # Move inputs to device
                inputs = inputs.to(device)

                # Run inputs through model
                with torch.no_grad():
                    outputs = model(**inputs)

                # Run pooling logic on outputs
                outputs = self.pooling(outputs, inputs["attention_mask"])

                # Add batch result
                results.extend(outputs.cpu().numpy())

            return np.asarray(results)

        # Build embeddings using sentence transformers
        return self.model.encode(documents, show_progress_bar=False)

    def chunk(self, texts, size):
        """
        Splits texts into separate batch sizes specified by size.

        Args:
            texts: text elements
            size: batch size

        Returns:
            list of evenly sized batches with the last batch having the remaining elements
        """

        return [texts[x : x + size] for x in range(0, len(texts), size)]

    def pooling(self, outputs, mask):
        """
        Runs mean pooling on token embeddings taking the input mask into account.

        Args:
            outputs: model outputs
            mask: input attention mask

        Returns:
            mean pooled vector using output token embeddings (i.e. last hidden state)
        """

        # pylint: disable=E1101
        mask = mask.unsqueeze(-1).expand(outputs[0].size()).float()
        return torch.sum(outputs[0] * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
