"""
TwelveLabs module
"""

# Conditional import
try:
    from twelvelabs import TwelveLabs as TwelveLabsAPI

    TWELVELABS = True
except ImportError:
    TWELVELABS = False

import os

from ...util import Library

from ..base import Vectors

# Core library imports
np = Library().numpy()


class TwelveLabs(Vectors):
    """
    Builds multimodal embeddings using the TwelveLabs Marengo model via the TwelveLabs API.

    Marengo generates embeddings in a shared space for text, images, audio and video, making it
    possible to run cross-modal similarity search (e.g. find video segments matching a text query).
    This backend supports text and url-based image/audio inputs. Each input is a string for text or
    a dict for other modalities, for example: {"image_url": "https://..."} or {"audio_url": "https://..."}.
    """

    @staticmethod
    def ismodel(path):
        """
        Checks if path is a TwelveLabs Marengo model.

        Args:
            path: input path

        Returns:
            True if this is a TwelveLabs Marengo model, False otherwise
        """

        return isinstance(path, str) and path.lower().startswith("marengo")

    def __init__(self, config, scoring, models):
        # Check before parent constructor since it calls loadmodel
        if not TWELVELABS:
            raise ImportError('TwelveLabs is not available - install "vectors" extra to enable')

        super().__init__(config, scoring, models)

    def loadmodel(self, path):
        # Build API client. API key resolved from config or the TWELVELABS_API_KEY environment variable.
        api = dict(self.config.get("vectors", {}).get("api", {}))
        api.setdefault("api_key", os.environ.get("TWELVELABS_API_KEY"))
        return TwelveLabsAPI(**api)

    def encode(self, data, category=None):
        # Model name (e.g. marengo3.0)
        model = self.config.get("path")

        # Optional create() parameters (excluding the api client settings)
        params = {k: v for k, v in self.config.get("vectors", {}).items() if k != "api"}

        # Embed each input - the Marengo embed API processes a single input per call
        embeddings = [self.embed(model, x, params) for x in data]

        return np.array(embeddings, dtype=np.float32)

    def embed(self, model, data, params):
        """
        Builds an embedding for a single input using the Marengo embed API.

        Args:
            model: Marengo model name
            data: input data - a string for text or a dict with one of image_url/audio_url for other modalities
            params: additional create() parameters

        Returns:
            embedding as a list of floats
        """

        # Build embed request - text input is the default, dicts route to other modalities
        request = data if isinstance(data, dict) else {"text": data}

        # Call the embed API
        response = self.model.embed.create(model_name=model, **request, **params)

        # Select the embedding for the requested modality
        embedding = response.text_embedding
        if "image_url" in request or "image_file" in request:
            embedding = response.image_embedding
        elif "audio_url" in request or "audio_file" in request:
            embedding = response.audio_embedding

        # Return the first segment's embedding vector
        return embedding.segments[0].float_
