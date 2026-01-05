"""
Pooling module
"""

import json

import numpy as np
import torch

from huggingface_hub.errors import HFValidationError
from torch import nn
from transformers.utils import cached_file

from ..models import Models


class Pooling(nn.Module):
    """
    Builds pooled vectors usings outputs from a transformers model.
    """

    def __init__(self, path, device, tokenizer=None, maxlength=None, loadprompts=None, modelargs=None):
        """
        Creates a new Pooling model.

        Args:
            path: path to model, accepts Hugging Face model hub id or local path
            device: tensor device id
            tokenizer: optional path to tokenizer
            maxlength: max sequence length
            loadprompts: whether instruction prompts should be loaded
            modelargs: additional model arguments
        """

        super().__init__()

        self.model = Models.load(path, modelargs=modelargs)
        self.tokenizer = Models.tokenizer(tokenizer if tokenizer else path)
        self.device = Models.device(device)

        # Detect unbounded tokenizer typically found in older models
        Models.checklength(self.model, self.tokenizer)

        # Set max length
        self.maxlength = maxlength if maxlength else self.tokenizer.model_max_length if self.tokenizer.model_max_length != int(1e30) else None

        # Load stored prompts
        self.prompts = self.loadprompts(path) if loadprompts else None

        # Move to device
        self.to(self.device)

    def encode(self, documents, batch=32, category=None):
        """
        Builds an array of pooled embeddings for documents.

        Args:
            documents: list of documents used to build embeddings
            batch: model batch size
            category: embeddings category (query or data)

        Returns:
            pooled embeddings
        """

        # Split documents into batches and process
        results = []

        # Apply pre encoding transformation logic
        documents = self.preencode(documents, category)

        # Sort document indices from largest to smallest to enable efficient batching
        # This performance tweak matches logic in sentence-transformers
        lengths = np.argsort([-len(x) if x else 0 for x in documents])
        documents = [documents[x] for x in lengths]

        for chunk in self.chunk(documents, batch):
            # Tokenize input
            inputs = self.tokenizer(chunk, padding=True, truncation="longest_first", return_tensors="pt", max_length=self.maxlength)

            # Move inputs to device
            inputs = inputs.to(self.device)

            # Run inputs through model
            with torch.no_grad():
                outputs = self.forward(**inputs)

            # Add batch result
            results.extend(outputs.cpu().to(torch.float32).numpy())

        # Apply post encoding transformation logic
        results = self.postencode(results, category)

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

    # pylint: disable=W0613
    def preencode(self, documents, category):
        """
        Applies pre encoding transformation logic.

        Args:
            documents: list of documents used to build embeddings
            category: embeddings category (query or data)
        """

        # Prepend prompt
        prompt = self.prompts.get(category) if self.prompts else None
        if prompt:
            documents = [f"{prompt}{x}" if isinstance(x, str) else x for x in documents]

        return documents

    # pylint: disable=W0613
    def postencode(self, results, category):
        """
        Applies post encoding transformation logic.

        Args:
            results: list of results
            category: embeddings category (query or data)

        Returns:
            results with transformation logic applied
        """

        return results

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

    def loadprompts(self, path):
        """
        Loads prompts from a sentence transformers configuration file.

        Args:
            path: model path

        Returns:
            prompts dictionary, if available
        """

        prompts = None
        config = self.load(path, "config_sentence_transformers.json")
        if config:
            # Copy document prompt to data
            prompts = config.get("prompts")
            if prompts and "document" in prompts:
                prompts["data"] = prompts["document"]

        return prompts
