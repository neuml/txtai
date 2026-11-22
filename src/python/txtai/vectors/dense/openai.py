"""
OpenAI module
"""

# Conditional import
try:
    import openai as api

    OPENAI = True
except ImportError:
    OPENAI = False

from ...util import Download, Library

from ..base import Vectors

# Core library imports
np = Library().numpy()


class OpenAI(Vectors):
    """
    Builds vectors using the OpenAI API or an OpenAI-compatible API endpoint.
    """

    # OpenAI client parameters, all other parameters are passed to the embeddings call
    CLIENT = ("api_key", "base_url", "organization", "project", "timeout", "max_retries", "default_headers", "default_query", "http_client")

    @staticmethod
    def ismodel(path):
        """
        Checks if path is an OpenAI model.

        Args:
            path: input path

        Returns:
            True if this is an OpenAI model, False otherwise
        """

        return isinstance(path, str) and OPENAI and path.lower().startswith("openai") and not OpenAI.ishub(path)

    @staticmethod
    def ishub(path):
        """
        Checks if path is available on the HF Hub.

        Args:
            input path

        Returns:
            True if this is a model on the HF Hub
        """

        # pylint: disable=W0702
        try:
            return Download()(path, "config.json") is not None if "/" in path else False
        except:
            return False

    def __init__(self, config, scoring, models):
        # Check before parent constructor since it calls loadmodel
        if not OPENAI:
            raise ImportError('OpenAI is not available - install "vectors" extra to enable')

        super().__init__(config, scoring, models)

    def loadmodel(self, path):
        # Accept api_base as an alias for base_url
        kwargs = {("base_url" if k == "api_base" else k): v for k, v in self.config.get("vectors", {}).items()}

        # Create OpenAI client
        return api.OpenAI(**{k: v for k, v in kwargs.items() if k in OpenAI.CLIENT})

    def encode(self, data, category=None):
        # Embeddings call parameters
        kwargs = {k: v for k, v in self.config.get("vectors", {}).items() if k not in OpenAI.CLIENT + ("api_base",)}

        # Remove openai provider prefix from model name
        path = self.config.get("path")
        path = path[len("openai/") :] if path.lower().startswith("openai/") else path

        # Call external embeddings API using the OpenAI client
        # Batching is handled server-side
        response = self.model.embeddings.create(model=path, input=data, **kwargs)

        # Read response into a NumPy array
        return np.array([x.embedding for x in response.data], dtype=np.float32)
