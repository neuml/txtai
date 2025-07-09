"""
Sparse module
"""

# Conditional import
try:
    from scipy.sparse import csr_matrix, vstack
    from sentence_transformers import SparseEncoder
    from sklearn.preprocessing import normalize

    SPARSE = True
except ImportError:
    SPARSE = False

from ..models import Models

from .base import Scoring
from .ivf import IVFFlat


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

        # Index backend
        self.backend = None

        # Input queue - data to be indexed
        self.queue = None

    def insert(self, documents, index=None):
        data = []
        for _, document, _ in documents:
            # Extract text, if necessary
            if isinstance(document, dict):
                document = document.get(self.text, document.get(self.object))

            if document is not None:
                # Add data
                data.append(" ".join(document) if isinstance(document, list) else document)

        # Encode and normalize data
        data = self.encode(data)

        # Create or add to existing sparse vectors
        self.queue = vstack((self.queue, data)) if self.queue is not None else data

    def delete(self, ids):
        self.backend.delete(ids)

    def index(self, documents=None):
        # Insert documents, if provided
        if documents:
            self.insert(documents)

        # Create index
        if self.queue is not None:
            # Create a new index instance
            self.backend = IVFFlat(self.config.get("ivf"))
            self.backend.index(self.queue)

        # Clear queue
        self.queue = None

    def upsert(self, documents=None):
        # Insert documents, if provided
        if documents:
            self.insert(documents)

        # Upsert index
        if self.backend and self.queue is not None:
            self.backend.append(self.queue)
        else:
            self.index()

        # Clear queue
        self.queue = None

    def weights(self, tokens):
        # Not supported
        return None

    def search(self, query, limit=3):
        return self.backend.search(self.encode([query]), limit)

    def batchsearch(self, queries, limit=3, threads=True):
        return [self.search(query, limit) for query in queries]

    def count(self):
        return self.backend.count()

    def load(self, path):
        # Read IVFFlat index
        self.backend = IVFFlat(self.config.get("ivf"))
        self.backend.load(path)

    def save(self, path):
        # Write IVFFlat index
        if self.backend is not None:
            self.backend.save(path)

    def close(self):
        # Close pool before model is closed
        if self.pool:
            self.model.stop_multi_process_pool(self.pool)
            self.pool = None

        self.model, self.backend, self.queue = None, None, None

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
        model = SparseEncoder(path, device=Models.device(deviceid), **modelargs)

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
            encoded data as a SciPy CSR Matrix
        """

        # Additional encoding arguments
        encodeargs = self.config.get("encodeargs", {})

        # Encode data
        data = self.model.encode(data, pool=self.pool, batch_size=self.encodebatch, **encodeargs)

        # Get data attributes
        data = data.cpu().coalesce()
        indices = data.indices().numpy()
        values = data.values().numpy()

        # Convert to CSR Matrix
        matrix = csr_matrix((values, indices), shape=data.size())

        # Normalize and return
        return normalize(matrix)
