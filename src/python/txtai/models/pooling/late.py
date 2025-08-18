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


class LatePooling(Pooling):
    """
    Builds late pooled vectors using outputs from a transformers model.
    """

    def __init__(self, path, device, tokenizer=None, maxlength=None, modelargs=None):
        super().__init__(path, device, tokenizer, maxlength, modelargs)

        # Get linear weights path
        config = self.load(path, "1_Dense/config.json")
        if config:
            # PyLate weights format
            name = "1_Dense/model.safetensors"
        else:
            # Stanford weights format
            name = "model.safetensors"

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

    def postencode(self, results):
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

        return np.asarray(data)

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
