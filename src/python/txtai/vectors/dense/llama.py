"""
Llama module
"""

import os

# Conditional import
try:
    import llama_cpp as llama

    LLAMA_CPP = True
except ImportError:
    LLAMA_CPP = False

from ...util import Download, Library

from ..base import Vectors

# Core library imports
np = Library().numpy()


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
        path = path if os.path.exists(path) else Download()(path)

        # Additional model arguments
        modelargs = self.config.get("vectors", {})

        # Default n_ctx to maxlength if available. Otherwise default n_ctx=0, which sets n_ctx=n_ctx_train.
        modelargs["n_ctx"] = modelargs.get("n_ctx", self.config.get("maxlength", 0))

        # Default n_batch to encode batch
        modelargs["n_batch"] = modelargs.get("n_batch", self.config.get("encodebatch", 64))

        # Default GPU layers if not already set
        modelargs["n_gpu_layers"] = modelargs.get("n_gpu_layers", -1 if self.config.get("gpu", os.environ.get("LLAMA_NO_METAL") != "1") else 0)

        # Default verbose flag
        modelargs["verbose"] = modelargs.get("verbose", False)

        # Create llama.cpp instance
        return llama.Llama(model_path=path, embedding=True, **modelargs)

    def encode(self, data, category=None):
        # Generate embeddings and return as a NumPy array
        # llama-cpp-python has it's own batching built-in using n_batch parameter
        return np.array(self.model.embed(data), dtype=np.float32)
