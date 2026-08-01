"""
Late module
"""

from .base import Pooling
from .lemur import Lemur
from .muvera import Muvera

# Core library imports
from ...util import Download, Library

library = Library()
np = library.numpy()
safetensors = library.safetensors()
torch = library.torch()
Module = library.module()


class LatePooling(Pooling):
    """
    Builds late pooled vectors using outputs from a transformers model.
    """

    def __init__(self, path, device, tokenizer=None, maxlength=None, loadprompts=None, modelargs=None):
        # Check if fixed dimensional encoder is enabled
        modelargs = modelargs.copy() if modelargs else {}
        muvera = modelargs.pop("muvera", {})
        lemur = modelargs.pop("lemur", None)
        centerconfigured = "center" in modelargs
        center = modelargs.pop("center", None)

        # Call parent initialization
        super().__init__(path, device, tokenizer, maxlength, loadprompts, modelargs)

        # Create fixed dimensional encoder
        self.encoder = Lemur(device=self.device, **lemur) if lemur is not None else Muvera(**muvera) if muvera is not None else None
        self.lengths = None

        # Get linear weights paths
        config = self.load(path, "modules.json")
        if config:
            # PyLate weights format
            models = [f"{x['path']}/model.safetensors" for x in config if x["path"].endswith("_Dense")]
        else:
            # Stanford weights format
            models = ["model.safetensors"]

        # Read model settings
        self.qprefix, self.qlength, self.dprefix, self.dlength = self.settings(path, config)

        # Load linear model
        self.linear = self.loadlinear(path, models)

        # Configure token centering
        self.center = self.centersettings(center, self.linear, centerconfigured)
        self.batches = None

    def forward(self, **inputs):
        """
        Runs late pooling on token embeddings.

        Args:
            inputs: model inputs

        Returns:
            Late pooled embeddings using output token embeddings (i.e. last hidden state)
        """

        # Track true token counts for centering and encoders that can't consume padding
        if self.center or getattr(self.encoder, "unpadded", False):
            self.lengths.extend(inputs["attention_mask"].sum(dim=1).cpu().tolist())

        # Track model batch boundaries for batch-scoped centering
        if self.center and self.center["scope"] == "batch":
            self.batches.append(inputs["attention_mask"].shape[0])

        # Run through transformers model
        tokens = super().forward(**inputs)

        # Run through final linear layer and return
        return self.linear(tokens)

    def preencode(self, documents, category):
        """
        Apply prefixes and lengths to data.

        Args:
            documents: list of documents used to build embeddings
            category: embeddings category (query or data)
        """

        # Reset true token counts for this encoding pass
        if self.center or getattr(self.encoder, "unpadded", False):
            self.lengths = []

        if self.center and self.center["scope"] == "batch":
            self.batches = []

        results = []

        # Apply prefix
        for text in documents:
            prefix = self.qprefix if category == "query" else self.dprefix
            if prefix:
                text = f"{prefix}{text}"

            results.append(text)

        # Set maxlength
        maxlength = self.qlength if category == "query" else self.dlength
        if maxlength:
            self.maxlength = maxlength

        return results

    def postencode(self, results, category):
        """
        Normalizes and pads results.

        Args:
            results: input results

        Returns:
            normalized results with padding
        """

        # LEMUR operates on normalized true token rows before batch padding
        if getattr(self.encoder, "unpadded", False):
            data = []
            for vectors, length in zip(results, self.lengths):
                vectors = vectors[:length]
                vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]
                data.append(vectors)

            if self.center:
                data = self.centerdata(data)

            return self.encoder(data, category)

        # Remove model padding before normalization. Padding is restored as exact zeros below.
        if self.center:
            results = [vectors[:length] for vectors, length in zip(results, self.lengths)]

        length = 0
        for vectors in results:
            # Get max length
            if vectors.shape[0] > length:
                length = vectors.shape[0]

            # Normalize vectors
            vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]

        # Pad values
        data = []
        for vectors in results:
            data.append(np.pad(vectors, [(0, length - vectors.shape[0]), (0, 0)]))

        # Build NumPy array
        data = np.asarray(data)

        # Center true token rows before fixed dimensional encoding
        if self.center:
            data = self.centerdata(data)

        # Apply fixed dimesional encoder, if necessary
        return self.encoder(data, category) if self.encoder else data

    @staticmethod
    def centersettings(center, linear, configured):
        """
        Validates and resolves token centering settings.

        Args:
            center: token centering configuration
            linear: loaded linear model
            configured: whether center was explicitly configured

        Returns:
            resolved token centering configuration
        """

        # Enable document centering by default for multi-linear models
        if not configured:
            center = sum(isinstance(module, torch.nn.Linear) for module in linear.modules()) > 1

        if isinstance(center, bool):
            return {"scope": "document"} if center else None

        if not isinstance(center, dict):
            raise ValueError("center must be a boolean or dictionary")

        center = center.copy()
        scope = center.pop("scope", "document")
        if scope not in ("document", "batch", "collection"):
            raise ValueError("center scope must be one of: document, batch, collection")

        mean = center.pop("mean", None)
        path = center.pop("path", None)
        if center:
            raise ValueError(f"unknown center setting: {next(iter(center))}")

        if scope != "collection" and (mean is not None or path is not None):
            raise ValueError("center mean and path are only valid with collection scope")

        if scope == "collection":
            if (mean is None) == (path is None):
                raise ValueError("collection center scope requires exactly one of mean or path")

            mean = np.load(path, allow_pickle=False) if path is not None else np.asarray(mean)
            if mean.ndim != 1 or not np.isfinite(mean).all():
                raise ValueError("collection center mean must be a finite one-dimensional array")

        return {"scope": scope, "mean": mean} if scope == "collection" else {"scope": scope}

    def centerdata(self, data):
        """
        Centers and re-normalizes true token rows.

        Args:
            data: normalized token vectors, with optional zero padding

        Returns:
            centered token vectors with zero padding preserved
        """

        array = isinstance(data, np.ndarray)
        vectors = [np.array(value, copy=True) for value in data]
        masks = [np.any(value != 0, axis=1) for value in vectors]
        scope = self.center["scope"]

        means = [None] * len(vectors)
        if scope == "batch":
            batches = getattr(self, "batches", None)
            sizes = batches if batches and sum(batches) == len(vectors) else [len(vectors)]
            offset = 0
            for size in sizes:
                rows = [vectors[x][masks[x]] for x in range(offset, offset + size) if masks[x].any()]
                mean = np.concatenate(rows).mean(axis=0) if rows else None
                means[offset : offset + size] = [mean] * size
                offset += size
        elif scope == "collection":
            means = [self.center["mean"]] * len(vectors)

        for x, (value, mask) in enumerate(zip(vectors, masks)):
            if not mask.any():
                continue

            current = value[mask]
            average = current.mean(axis=0) if scope == "document" else means[x]
            if average is not None and average.shape != (current.shape[1],):
                raise ValueError("center mean dimension must match token vector dimension")

            current = current - average
            norms = np.linalg.norm(current, axis=1, keepdims=True)
            current = np.divide(current, norms, out=np.zeros_like(current), where=norms != 0)
            vectors[x][mask] = current

        return np.asarray(vectors) if array else vectors

    def settings(self, path, config):
        """
        Reads model settings.

        Args:
            path: model path
            config: PyLate model format if provided, otherwise read from Stanford format
        """

        if config:
            # PyLate format
            config = self.load(path, "config_sentence_transformers.json")
            params = ["query_prefix", "query_length", "document_prefix", "document_length"]
        else:
            # Stanford format
            config = self.load(path, "artifact.metadata")
            params = ["query_token_id", "query_maxlen", "doc_token_id", "doc_maxlen"]

        return [config.get(p) for p in params]

    def loadlinear(self, path, models):
        """
        Loads linear model.

        Args:
            path: model path
            models: list of paths to each dense model

        Returns:
            linear model
        """

        # Load dense layers as a sequential model
        layers = []
        for model in models:
            model = Download()(path, model)
            with safetensors.safe_open(filename=model, framework="pt") as f:
                dense = []
                for name in ["linear.weight", "residual.weight"]:
                    if name in f.keys():
                        weights = f.get_tensor(name)

                        # Load weights into linear layer
                        model = torch.nn.Linear(weights.shape[1], weights.shape[0], bias=False, device=self.device, dtype=weights.dtype)
                        with torch.no_grad():
                            model.weight.copy_(weights)

                        dense.append(model)

                layers.append(Dense(*dense))

        return torch.nn.Sequential(*layers)


class Dense(Module):
    """
    Dense layer. Supports multiple linear layers that sum into a final answer.
    """

    def __init__(self, *modules):
        """
        Create a Dense layer.

        Args:
            modules: list of modules to sum outputs
        """

        super().__init__()
        self.layers = torch.nn.ModuleList(modules)

    def forward(self, x):
        """
        Sums the outputs of each module for the input.

        Args:
            x: input

        Returns:
            sum of the outputs of each layer
        """

        # Compute sum of all module outputs
        return sum(layer(x) for layer in self.layers)
