"""
Max module
"""

from .base import Pooling

# Core library imports
from ...util import Library

torch = Library().torch()


class MaxPooling(Pooling):
    """
    Builds max pooled vectors usings outputs from a transformers model.
    """

    def forward(self, **inputs):
        """
        Runs max pooling on token embeddings taking the input mask into account.

        Args:
            inputs: model inputs

        Returns:
            max pooled embeddings using output token embeddings (i.e. last hidden state)
        """

        # Run through transformers model
        tokens = super().forward(**inputs)
        mask = inputs["attention_mask"]

        # Max pooling
        # pylint: disable=E1101
        mask = mask.unsqueeze(-1).expand(tokens.size()).bool()
        return torch.max(tokens.masked_fill(~mask, torch.finfo(tokens.dtype).min), 1).values
