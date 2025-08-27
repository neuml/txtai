"""
Late module
"""

import json

import numpy as np
import torch

from huggingface_hub.errors import HFValidationError
from safetensors import safe_open
from torch import nn
from transformers.utils import cached_file

from .base import Pooling
from .muvera import Muvera


class LatePooling(Pooling):
    """
    Builds late pooled vectors using outputs from a transformers model.
    """

    def __init__(self, path, device, tokenizer=None, maxlength=None, modelargs=None):
        # Check if fixed dimensional encoder is enabled
        modelargs = modelargs.copy() if modelargs else {}
        muvera = modelargs.pop("muvera", {})
        self.encoder = Muvera(**muvera) if muvera is not None else None

        # Call parent initialization
        super().__init__(path, device, tokenizer, maxlength, modelargs)

        # Get linear weights path
        config = self.load(path, "1_Dense/config.json")
        if config:
            # PyLate weights format
            name = "1_Dense/model.safetensors"
        else:
            # Stanford weights format
            name = "model.safetensors"

        # Read model settings
        self.qprefix, self.qlength, self.dprefix, self.dlength = self.settings(path, config)

        # Load linear layer
        path = cached_file(path_or_repo_id=path, filename=name)
        with safe_open(filename=path, framework="pt") as f:
            weights = f.get_tensor("linear.weight")

            # Load weights into linear layer
            self.linear = nn.Linear(weights.shape[1], weights.shape[0], bias=False, device=self.device, dtype=weights.dtype)
            with torch.no_grad():
                self.linear.weight.copy_(weights)

    def forward(self, **inputs):
        """
        Runs late pooling on token embeddings.

        Args:
            inputs: model inputs

        Returns:
            Late pooled embeddings using output token embeddings (i.e. last hidden state)
        """

        # Run through transformers model
        tokens = super().forward(**inputs)

        # Run through final linear layer and return
        return self.linear(tokens)

    def preencode(self, documents, category):
        """
        Apply prefixes and lengths to data.

        Args:
            documents: list of documents used to build embeddings
            category: embeddings category (query or data)
        """

        results = []

        # Apply prefix
        for text in documents:
            prefix = self.qprefix if category == "query" else self.dprefix
            if prefix:
                text = f"{prefix}{text}"

            results.append(text)

        # Set maxlength
        maxlength = self.qlength if category == "query" else self.dlength
        if maxlength:
            self.maxlength = maxlength

        return results

    def postencode(self, results, category):
        """
        Normalizes and pads results.

        Args:
            results: input results

        Returns:
            normalized results with padding
        """

        length = 0
        for vectors in results:
            # Get max length
            if vectors.shape[0] > length:
                length = vectors.shape[0]

            # Normalize vectors
            vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]

        # Pad values
        data = []
        for vectors in results:
            data.append(np.pad(vectors, [(0, length - vectors.shape[0]), (0, 0)]))

        # Build NumPy array
        data = np.asarray(data)

        # Apply fixed dimesional encoder, if necessary
        return self.encoder(data, category) if self.encoder else data

    def settings(self, path, config):
        """
        Reads model settings.

        Args:
            path: model path
            config: PyLate model format if provided, otherwise read from Stanford format
        """

        if config:
            # PyLate format
            config = self.load(path, "config_sentence_transformers.json")
            params = ["query_prefix", "query_length", "document_prefix", "document_length"]
        else:
            # Stanford format
            config = self.load(path, "artifact.metadata")
            params = ["query_token_id", "query_maxlen", "doc_token_id", "doc_maxlen"]

        return [config.get(p) for p in params]

    def load(self, path, name):
        """
        Loads a JSON config file from the Hugging Face Hub.

        Args:
            path: model path
            name: file to load

        Returns:
            config
        """

        # Download file and parse JSON
        config = None
        try:
            path = cached_file(path_or_repo_id=path, filename=name)
            if path:
                with open(path, encoding="utf-8") as f:
                    config = json.load(f)

        # Ignore this error - invalid repo or directory
        except (HFValidationError, OSError):
            pass

        return config
