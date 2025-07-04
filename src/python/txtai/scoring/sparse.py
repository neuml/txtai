"""
Sparse module
"""

import numpy as np
import torch

# Conditional import
try:
    from sentence_transformers import SparseEncoder

    SPARSE = True
except ImportError:
    SPARSE = False

from ..models import Models

from .base import Scoring


class Sparse(Scoring):
    """
    Sparse vector scoring.
    """

    def __init__(self, config=None, models=None):
        super().__init__(config)

        if not SPARSE:
            raise ImportError('Sparse encoder is not available - install "scoring" extra to enable')

        # Models cache
        self.models = models

        # Pool parameter
        self.pool = None

        # Sparse encoder
        self.model = self.loadmodel()

        # Encode batch size - controls underlying model batch size when encoding vectors
        self.encodebatch = config.get("encodebatch", 32)

        # Sparse data backend
        self.backend = None

        # Deleted rows
        self.deletes = []

    def insert(self, documents, index=None):
        data = []
        for _, document, _ in documents:
            # Extract text, if necessary
            if isinstance(document, dict):
                document = document.get(self.text, document.get(self.object))

            if document is not None:
                # Add data
                data.append(" ".join(document) if isinstance(document, list) else document)

        # Encode batch and append to data
        data = self.encode(data)
        self.backend = torch.cat((self.backend, data)) if self.backend is not None else data

    def delete(self, ids):
        self.deletes.append(ids)

    def weights(self, tokens):
        # Not supported
        return None

    def search(self, query, limit=3):
        return self.batchsearch([query], limit)[0]

    def batchsearch(self, queries, limit=3, threads=True):
        queries = self.encode(queries)
        scores = self.model.similarity(queries, self.backend).cpu().numpy()

        # Clear deletes
        scores[:, self.deletes] = 0

        # Get top n scores
        indices = np.argpartition(-scores, limit if limit < scores.shape[0] else scores.shape[0] - 1)[:, :limit]
        scores = np.clip(np.take_along_axis(scores, indices, axis=1), 0.0, 1.0)

        # Get top n results
        results = []
        for x, index in enumerate(indices):
            results.append(list(zip(index.tolist(), scores[x].tolist())))

        return results

    def count(self):
        return self.backend.shape[0] - len(self.deletes) if self.backend is not None else 0

    def load(self, path):
        with open(path, "rb") as f:
            # Load sparse index
            indices, values, size = np.load(f), np.load(f), np.load(f)

            # Load deletes
            self.deletes = np.load(f).tolist()

        # Create backend - load on same device as model
        self.backend = torch.sparse_coo_tensor(indices, values, size=torch.Size(size), device=self.model.device)

    def save(self, path):
        if self.backend is not None:
            with open(path, "wb") as f:
                # Save sparse index
                data = self.backend.coalesce().cpu()
                np.save(f, data.indices())
                np.save(f, data.values())
                np.save(f, data.size())

                # Save deletes
                np.save(f, np.array(self.deletes))

    def close(self):
        # Close pool before model is closed
        if self.pool:
            self.model.stop_multi_process_pool(self.pool)
            self.pool = None

        self.model, self.backend = None, None

    def hasterms(self):
        return True

    def isnormalized(self):
        return True

    def loadmodel(self):
        """
        Loads the sparse encoder model.

        Returns:
            SparseEncoder
        """

        # Model path
        path = self.config.get("path")

        # Check if model is cached
        if self.models and path in self.models:
            return self.models[path]

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
        modelargs = self.config.get("modelargs", {})

        # Build embeddings with sentence-transformers
        model = SparseEncoder(path, device=Models.device(deviceid), similarity_fn_name="cosine", **modelargs)

        # Start process pool for multiple GPUs
        if pool:
            self.pool = model.start_multi_process_pool()

        # Store model in cache
        if self.models is not None and path:
            self.models[path] = model

        # Return model
        return model

    def encode(self, data):
        """
        Encodes a batch of data using the Sparse Encoder model.

        Args:
            data: input data

        Returns:
            encoded data
        """

        # Additional encoding arguments
        encodeargs = self.config.get("encodeargs", {})

        # Encode data
        return self.model.encode(data, pool=self.pool, batch_size=self.encodebatch, **encodeargs)
