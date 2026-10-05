"""
Last module
"""

from .base import Pooling

# Core library imports
from ...util import Library

torch = Library().torch()


class LastPooling(Pooling):
    """
    Builds last token pooled vectors usings outputs from a transformers model.
    """

    def forward(self, **inputs):
        """
        Runs last pooling on token embeddings taking the input mask into account.

        Args:
            inputs: model inputs

        Returns:
            last pooled embeddings using output token embeddings (i.e. last hidden state)
        """

        # Run through transformers model
        tokens = super().forward(**inputs)
        mask = inputs["attention_mask"]

        # Last pooling logic from Sentence Transformers
        _, sequence, dimensions = tokens.shape

        # Avoid tracing the argmax with int64 input that can not be handled by ONNX Runtime
        mask = mask.to(torch.int32) if torch.jit.is_tracing() else mask

        # Use flip and max() to get the last index of 1 in the attention mask
        values, indices = mask.flip(1).max(1)
        indices = torch.where(values == 0, sequence - 1, indices)
        gather = sequence - indices - 1

        # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
        gather = gather.unsqueeze(-1).repeat(1, dimensions)
        gather = gather.unsqueeze(1)

        # Expand mask to ignore 0 index attention masks
        mask = mask.unsqueeze(-1).expand(tokens.size()).to(tokens.dtype)

        # Return last pooled embeddings
        return torch.gather(tokens * mask, 1, gather).squeeze(dim=1)
