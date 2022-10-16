"""
Tensor processing framework module
"""

import torch

from .base import Pipeline


class Tensors(Pipeline):
    """
    Pipeline backed by a tensor processing framework. Currently supports PyTorch.
    """

    def quantize(self, model):
        """
        Quantizes input model and returns. This only is supported for CPU devices.

        Args:
            model: torch model

        Returns:
            quantized torch model
        """

        # pylint: disable=E1101
        return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    def tensor(self, data):
        """
        Creates a tensor array.

        Args:
            data: input data

        Returns:
            tensor
        """

        # pylint: disable=E1102
        return torch.tensor(data)

    def context(self):
        """
        Defines a context used to wrap processing with the tensor processing framework.

        Returns:
            processing context
        """

        # pylint: disable=E1101
        return torch.no_grad()
