"""
Sentence Transformers module
"""

# Conditional import
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS = True
except ImportError:
    SENTENCE_TRANSFORMERS = False

from ...models import Models

from ..base import Vectors


class STVectors(Vectors):
    """
    Builds vectors using sentence-transformers (aka SBERT).
    """

    def __init__(self, config, scoring, models):
        # Check before parent constructor since it calls loadmodel
        if not SENTENCE_TRANSFORMERS:
            raise ImportError('sentence-transformers is not available - install "vectors" extra to enable')

        # Pool parameter created here since loadmodel is called from parent constructor
        self.pool = None

        super().__init__(config, scoring, models)

    def loadmodel(self, path):
        # Get target device
        gpu, pool = self.config.get("gpu", True), False

        # Default mode uses a single GPU. Setting to all spawns a process per GPU.
        if isinstance(gpu, str) and gpu == "all":
            # Get number of accelerator devices available
            devices = Models.acceleratorcount()

            # Enable multiprocessing pooling only when multiple devices are available
            gpu, pool = devices <= 1, devices > 1

        # Tensor device id
        deviceid = Models.deviceid(gpu)

        # Additional model arguments
        modelargs = self.config.get("vectors", {})

        # Load sentence-transformers encoder
        model = self.loadencoder(path, device=Models.device(deviceid), **modelargs)

        # Start process pool for multiple GPUs
        if pool:
            self.pool = model.start_multi_process_pool()

        # Return model
        return model

    def encode(self, data, category=None):
        # Get encode method based on input category
        encode = self.model.encode_query if category == "query" else self.model.encode_document if category == "data" else self.model.encode

        # Additional encoding arguments
        encodeargs = self.config.get("encodeargs", {})

        # Encode with sentence transformers encoder
        return encode(data, pool=self.pool, batch_size=self.encodebatch, **encodeargs)

    def close(self):
        # Close pool before model is closed in parent method
        if self.pool:
            self.model.stop_multi_process_pool(self.pool)
            self.pool = None

        super().close()

    def loadencoder(self, path, device, **kwargs):
        """
        Loads the embeddings encoder model from path.

        Args:
            path: model path
            device: tensor device
            kwargs: additional keyword args

        Returns:
            embeddings encoder
        """

        return SentenceTransformer(path, device=device, **kwargs)
