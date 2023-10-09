"""
Mean module
"""

import torch

from .base import Pooling


class MeanPooling(Pooling):
    """
    Builds mean pooled vectors usings outputs from a transformers model.
    """

    def forward(self, **inputs):
        """
        Runs mean pooling on token embeddings taking the input mask into account.

        Args:
            inputs: model inputs

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
