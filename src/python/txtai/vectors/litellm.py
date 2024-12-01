"""
LiteLLM module
"""

import numpy as np

# Conditional import
try:
    import litellm as api

    LITELLM = True
except ImportError:
    LITELLM = False

from .base import Vectors


class LiteLLM(Vectors):
    """
    Builds vectors using an external embeddings API via LiteLLM.
    """

    @staticmethod
    def ismodel(path):
        """
        Checks if path is a LiteLLM model.

        Args:
            path: input path

        Returns:
            True if this is a LiteLLM model, False otherwise
        """

        # pylint: disable=W0702
        if isinstance(path, str) and LITELLM:
            debug = api.suppress_debug_info
            try:
                # Suppress debug messages for this test
                api.suppress_debug_info = True
                return api.get_llm_provider(path)
            except:
                return False
            finally:
                # Restore debug info value to original value
                api.suppress_debug_info = debug

        return False

    def __init__(self, config, scoring, models):
        # Check before parent constructor since it calls loadmodel
        if not LITELLM:
            raise ImportError('LiteLLM is not available - install "vectors" extra to enable')

        super().__init__(config, scoring, models)

    def loadmodel(self, path):
        return None

    def encode(self, data):
        # Call external embeddings API using LiteLLM
        # Batching is handled server-side
        response = api.embedding(model=self.config.get("path"), input=data, **self.config.get("vectors", {}))

        # Read response into a NumPy array
        return np.array([x["embedding"] for x in response.data], dtype=np.float32)
