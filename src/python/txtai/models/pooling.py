"""
Pooling module
"""

import numpy as np
import torch

from torch import nn

from transformers import AutoTokenizer

from .models import Models


class Pooling(nn.Module):
    """
    Builds pooled vectors usings outputs from a transformers model.
    """

    def __init__(self, path, device, tokenizer=None, batch=32, maxlength=None):
        """
        Creates a new Pooling model.

        Args:
            path: path to model, accepts Hugging Face model hub id or local path
            device: tensor device id
            tokenizer: optional path to tokenizer
            batch: batch size
            maxlength: max sequence length
        """

        super().__init__()

        self.model = Models.load(path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer if tokenizer else path)
        self.device = Models.device(device)

        # Detect unbounded tokenizer typically found in older models
        Models.checklength(self.model, self.tokenizer)

        # Set batch and max length
        self.batch = batch
        self.maxlength = maxlength if maxlength else self.tokenizer.model_max_length

        # Move to device
        self.to(self.device)

    def encode(self, documents):
        """
        Builds an array of pooled embeddings for documents.

        Args:
            documents: list of documents used to build embeddings

        Returns:
            pooled embeddings
        """

        # Split documents into batches and process
        results = []

        # Sort document indices from largest to smallest to enable efficient batching
        # This performance tweak matches logic in sentence-transformers
        lengths = np.argsort([-len(x) for x in documents])
        documents = [documents[x] for x in lengths]

        for chunk in self.chunk(documents, self.batch):
            # Tokenize input
            inputs = self.tokenizer(chunk, padding=True, truncation="longest_first", return_tensors="pt", max_length=self.maxlength)

            # Move inputs to device
            inputs = inputs.to(self.device)

            # Run inputs through model
            with torch.no_grad():
                outputs = self.forward(**inputs)

            # Add batch result
            results.extend(outputs.cpu().numpy())

        # Restore original order and return array
        return np.asarray([results[x] for x in np.argsort(lengths)])

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

    def forward(self, **inputs):
        """
        Runs inputs through transformers model and returns outputs.

        Args:
            inputs: model inputs

        Returns:
            model outputs
        """

        return self.model(**inputs)[0]


class MeanPooling(Pooling):
    """
    Builds mean pooled vectors usings outputs from a transformers model.
    """

    def forward(self, **inputs):
        """
        Runs mean pooling on token embeddings taking the input mask into account.

        Args:
            outputs: model outputs
            mask: input attention mask

        Returns:
            mean pooled embeddings using output token embeddings (i.e. last hidden state)
        """

        # Run through transformers model
        tokens = super().forward(**inputs)
        mask = inputs["attention_mask"]

        # Mean pooling
        # pylint: disable=E1101
        mask = mask.unsqueeze(-1).expand(tokens.size()).float()
        return torch.sum(tokens * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
