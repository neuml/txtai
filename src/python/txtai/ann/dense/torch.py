"""
PyTorch module
"""

import numpy as np
import torch

from .numpy import NumPy


class Torch(NumPy):
    """
    Builds an ANN index backed by a PyTorch array.
    """

    def __init__(self, config):
        super().__init__(config)

        # Define array functions
        self.all, self.cat, self.dot, self.zeros = torch.all, torch.cat, torch.mm, torch.zeros
        self.argsort, self.xor, self.clip = torch.argsort, torch.bitwise_xor, torch.clip

    def tensor(self, array):
        # Convert array to Tensor
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)

        # Load to GPU device, if available
        return array.cuda() if torch.cuda.is_available() else array

    def numpy(self, array):
        return array.cpu().numpy()

    def totype(self, array, dtype):
        return array.long() if dtype == np.int64 else array

    def settings(self):
        return {"torch": torch.__version__}
