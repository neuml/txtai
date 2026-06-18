"""
Late module
"""

from .base import Pooling
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
        self.encoder = Muvera(**muvera) if muvera is not None else None

        # Call parent initialization
        super().__init__(path, device, tokenizer, maxlength, loadprompts, modelargs)

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

    def forward(self, **inputs):
        """
        Runs late pooling on token embeddings.

        Args:
            inputs: model inputs

        Returns:
            Late pooled embeddings using output token embeddings (i.e. last hidden state)
        """

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

        # Apply fixed dimesional encoder, if necessary
        return self.encoder(data, category) if self.encoder else data

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
