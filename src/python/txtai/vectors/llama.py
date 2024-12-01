"""
Llama module
"""

import os

import numpy as np

from huggingface_hub import hf_hub_download

# Conditional import
try:
    from llama_cpp import Llama

    LLAMA_CPP = True
except ImportError:
    LLAMA_CPP = False

from .base import Vectors


class LlamaCpp(Vectors):
    """
    Builds vectors using llama.cpp.
    """

    @staticmethod
    def ismodel(path):
        """
        Checks if path is a llama.cpp model.

        Args:
            path: input path

        Returns:
            True if this is a llama.cpp model, False otherwise
        """

        return isinstance(path, str) and path.lower().endswith(".gguf")

    def __init__(self, config, scoring, models):
        # Check before parent constructor since it calls loadmodel
        if not LLAMA_CPP:
            raise ImportError('llama.cpp is not available - install "vectors" extra to enable')

        super().__init__(config, scoring, models)

    def loadmodel(self, path):
        # Check if this is a local path, otherwise download from the HF Hub
        path = path if os.path.exists(path) else self.download(path)

        # Additional model arguments
        modelargs = self.config.get("vectors", {})

        # Default GPU layers if not already set
        modelargs["n_gpu_layers"] = modelargs.get("n_gpu_layers", -1 if self.config.get("gpu", os.environ.get("LLAMA_NO_METAL") != "1") else 0)

        # Create llama.cpp instance
        return Llama(path, n_ctx=0, verbose=modelargs.pop("verbose", False), embedding=True, **modelargs)

    def encode(self, data):
        # Generate embeddings and return as a NumPy array
        # llama-cpp-python has it's own batching built-in using n_batch parameter
        return np.array(self.model.embed(data), dtype=np.float32)

    def download(self, path):
        """
        Downloads path from the Hugging Face Hub.

        Args:
            path: full model path

        Returns:
            local cached model path
        """

        # Split into parts
        parts = path.split("/")

        # Calculate repo id split
        repo = 2 if len(parts) > 2 else 1

        # Download and cache file
        return hf_hub_download(repo_id="/".join(parts[:repo]), filename="/".join(parts[repo:]))
