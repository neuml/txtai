"""
PyTorch module
"""

import numpy as np
import torch

try:
    from bitsandbytes import matmul_4bit
    from bitsandbytes.functional import (
        QuantState,
        int8_vectorwise_quant,
        int8_vectorwise_dequant,
        int8_linear_matmul,
        int8_mm_dequant,
        quantize_4bit,
        dequantize_4bit,
    )

    BNB = True
except ImportError:
    BNB = False

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

        # Quantization parameters
        self.qstate, self.qdeleted = None, 0

        # Initialize quantization
        settings = self.qsettings()
        if settings:
            if not BNB:
                raise ImportError('bitsandbytes is not available - install "ann" extra to enable')

            if settings.get("type") == "int8":
                # Matrix multiply for 8 bit vectors
                self.dot = self.matmul8bit
            else:
                # Matrix multiply for 4 bit vectors
                self.dot = self.matmul4bit

            # Require safetensors storage
            self.config[self.config["backend"]]["safetensors"] = True

    def index(self, embeddings):
        with QuantizeContext(self):
            super().index(embeddings)

    def append(self, embeddings):
        with QuantizeContext(self):
            super().append(embeddings)

    def delete(self, ids):
        with QuantizeContext(self):
            super().delete(ids)

            # Calculate deleted for quantized data, if necessary
            if self.qstate:
                self.qdeleted = self.qstate.shape[0] - super().count()

    def count(self):
        return self.qstate.shape[0] - self.qdeleted if self.qstate else super().count()

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

    def loadsafetensors(self, path):
        data = super().loadsafetensors(path)

        # Load quantization settings
        if self.qsettings():
            self.qstate = QuantState(
                absmax=self.tensor(data["absmax"]),
                shape=torch.Size(data["shape"].tolist()),
                code=self.tensor(data["code"]) if "code" in data else None,
                blocksize=int(data["blocksize"]) if "blocksize" in data else None,
                quant_type=data["quant_type"],
                dtype=getattr(torch, data["dtype"]),
            )
            self.qdeleted = int(data["qdeleted"])

        return data

    def savesafetensors(self, data, path, metadata=None):
        # Save quantization settings
        if self.qstate:
            # Required elements
            data["absmax"] = self.qstate.absmax.cpu().numpy()
            data["shape"] = np.array(list(self.qstate.shape))

            metadata = {
                "quant_type": str(self.qstate.quant_type),
                "dtype": str(self.qstate.dtype).rsplit(".", maxsplit=1)[-1],
                "qdeleted": str(self.qdeleted),
            }

            # Add optional elements
            if self.qstate.code is not None:
                data["code"] = self.qstate.code.cpu().numpy()

            if self.qstate.blocksize:
                metadata["blocksize"] = str(self.qstate.blocksize)

        super().savesafetensors(data, path, metadata)

    def quantize(self):
        """
        Quantizes data if quantization if supported and enabled.
        """

        # Get quantization settings and quantize
        settings = self.qsettings()
        if settings:
            if settings.get("type") == "int8":
                # Get current backend config
                shape, dtype = self.backend.shape, self.backend.dtype

                # 8-bit quantization
                self.backend, absmax, _ = int8_vectorwise_quant(self.backend.half())
                self.qstate = QuantState(absmax=absmax, shape=shape, quant_type=settings["type"], dtype=dtype)
            else:
                # 4-bit quantization
                self.backend, self.qstate = quantize_4bit(
                    self.backend, blocksize=settings.get("blocksize", 64), quant_type=settings.get("type", "nf4")
                )

    def dequantize(self):
        """
        Dequantizes data if quantization is supported and enabled.
        """

        # Dequantize using current quantization state
        if self.qstate:
            if self.qstate.quant_type == "int8":
                # 8-bit quantization
                self.backend = int8_vectorwise_dequant(self.backend, self.qstate.absmax)
            else:
                # 4-bit quantization
                self.backend = dequantize_4bit(self.backend, self.qstate)

    def qsettings(self):
        """
        Parse quantization settings. Only read parameters if CUDA is available.

        Returns:
            {quantization settings}
        """

        quantize = self.setting("quantize")
        return {"quantize": True} if quantize and isinstance(quantize, bool) else quantize

    def matmul8bit(self, query, data):
        """
        8-bit integer matrix multiplication.

        Args:
            query: query matrix
            data: data matrix

        Returns:
            query @ data
        """

        # Matrix multiplication method requires transposing data matrix
        query, absmax, _ = int8_vectorwise_quant(query.half())
        return int8_mm_dequant(int8_linear_matmul(query, data.T), absmax, self.qstate.absmax).float()

    def matmul4bit(self, query, data):
        """
        4-bit float matrix multiplication.

        Args:
            query: query matrix
            data: data matrix

        Returns:
            query @ data
        """

        # Matrix multiplication method transposes data already
        return matmul_4bit(query, data, self.qstate)


class QuantizeContext:
    """
    Quantization context. Facilitates modifications to quantized tensors.
    """

    def __init__(self, ann):
        self.ann = ann

    def __enter__(self):
        self.ann.dequantize()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ann.quantize()
