"""
LEMUR module
"""

import json
import math
import os

# Core library imports
from ...util import Download, Library

library = Library()
np = library.numpy()
safetensors = library.safetensors()
torch = library.torch()
Module = library.module()


class Lemur:
    """
    Implements the LEMUR (Learned Multi-Vector Retrieval) algorithm. This reduces
    late interaction multi-vector outputs to a single fixed vector.

    LEMUR artifacts contain a feature encoder, output normalization statistics and
    a sample of learn-token vectors used to solve document weights. Query vectors
    are summed feature encodings while document vectors are ordinary least squares
    weights over the stored sample.

    This code is based on the following:
      - Paper: https://arxiv.org/abs/2601.21853
      - GitHub: https://github.com/ejaasaari/lemur
    """

    # Signals LatePooling that this encoder must receive true, unpadded token rows
    unpadded = True

    def __init__(self, path=None, device=None):
        """
        Creates a LEMUR encoder.

        Args:
            path: local artifact directory or Hugging Face Hub path
            device: tensor device
        """

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model = None
        self.sample = None
        self.mean = None
        self.std = None
        self.config = None
        self.pinv = None
        self.selected_epoch = None
        self.selected_loss = None
        self.selection_metric = None

        if path:
            self.load(path)

    def __call__(self, data, category):
        """
        Transforms multi-vector collections into fixed dimensional vectors.

        Args:
            data: list or array of unpadded multi-vector collections
            category: embeddings category (query or data)

        Returns:
            fixed dimensional vectors
        """

        if category == "query":
            vectors = self.compute_features(data)
        elif category == "data":
            vectors = self.compute_weights(data)
        else:
            raise ValueError("category must be query or data")

        return vectors.detach().cpu().to(torch.float32).numpy()

    # pylint: disable=R0913,R0917
    def fit(
        self,
        data,
        output=None,
        train_subset_size=8192,
        learn_subset_size=100000,
        ols_sample_size=16384,
        num_layers=1,
        hidden_dim=512,
        final_hidden_dim=2048,
        activation="gelu",
        epochs=None,
        lr=3e-3,
        batch_size=512,
        grad_clip=0.5,
        query_scale=32,
        seed=42,
        validation_split=0.0,
        learn=None,
    ):
        """
        Fits a LEMUR feature encoder and OLS sample.

        Args:
            data: iterable of unpadded corpus token-vector arrays
            output: optional artifact output directory
            train_subset_size: maximum number of documents used to build targets
            learn_subset_size: maximum number of token vectors used to learn features
            ols_sample_size: maximum number of token vectors stored for document weights
            num_layers: number of MLP feature layers
            hidden_dim: MLP hidden dimension
            final_hidden_dim: fixed output dimension
            activation: feature activation
            epochs: required training choice; 100 is the quality MLP setting and 0 selects deterministic ELM features
            lr: Adam learning rate
            batch_size: training batch size
            grad_clip: optional gradient clipping value
            query_scale: query feature sum scale
            seed: random seed
            validation_split: fraction of sampled learn tokens held out for validation
            learn: optional iterable of learn-token arrays, defaults to data

        Returns:
            self
        """

        if epochs is None:
            raise ValueError("epochs must be set explicitly: use epochs=100 for trained MLP quality or epochs=0 for deterministic ELM features")
        if epochs < 0:
            raise ValueError("epochs must be greater than or equal to 0")
        for name, value in [
            ("ols_sample_size", ols_sample_size),
            ("query_scale", query_scale),
            ("num_layers", num_layers),
            ("batch_size", batch_size),
            ("train_subset_size", train_subset_size),
            ("learn_subset_size", learn_subset_size),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0")
        if final_hidden_dim < 1:
            raise ValueError("final_hidden_dim must be greater than 0")
        if validation_split < 0 or validation_split >= 1:
            raise ValueError("validation_split must be greater than or equal to 0 and less than 1")

        self.selected_epoch = None
        self.selected_loss = None
        self.selection_metric = None

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        training = self._create_training_data(
            data,
            learn,
            (train_subset_size, learn_subset_size),
            validation_split,
            generator,
        )

        model_type = "elm" if epochs == 0 else "mlp"
        config = {
            "model_type": model_type,
            "input_dim": training["dimension"],
            "hidden_dim": hidden_dim,
            "final_hidden_dim": final_hidden_dim,
            "num_layers": num_layers,
            "activation": activation,
            "query_scale": query_scale,
            "seed": seed,
        }

        # Isolate seeded model construction from the caller's global RNG state
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = LemurModel(output_dim=training["train_size"], **config).to(self.device)

        if epochs:
            self.train(
                training["inputs"],
                training["targets"],
                epochs,
                lr,
                batch_size,
                grad_clip,
                seed,
                training["validation_inputs"],
                training["validation_targets"],
            )
            config["training"] = {
                "selected_epoch": self.selected_epoch,
                "selected_loss": self.selected_loss,
                "selection_metric": self.selection_metric,
                "validation_split": validation_split,
            }

        # Persist the exact learn-token sample used by future document upserts
        learn_tokens = training["learn"]
        sample_size = min(ols_sample_size, len(learn_tokens))
        sample_indices = torch.randperm(len(learn_tokens), generator=generator)[:sample_size].to(learn_tokens.device)
        self.sample = learn_tokens[sample_indices].detach()
        self.config = config
        self.pinv = None

        if output:
            self.save(output)

        return self

    def _create_training_data(self, data, learn, subset_sizes, validation_split, generator):
        """
        Creates sampled training and optional validation data.

        Args:
            data: iterable of unpadded corpus token-vector arrays
            learn: optional iterable of learn-token arrays
            subset_sizes: train-document and learn-token sample limits
            validation_split: fraction of sampled learn tokens held out for validation
            generator: seeded random generator

        Returns:
            sampled training data
        """

        documents = self.documents(data)
        if not documents:
            raise ValueError("data must contain at least one document")

        dimension = documents[0].shape[1]
        if any(document.shape[1] != dimension for document in documents):
            raise ValueError("all token vectors must have the same dimension")

        learn_documents = documents if learn is None else self.documents(learn)
        if not learn_documents:
            raise ValueError("learn must contain at least one document")
        if any(document.shape[1] != dimension for document in learn_documents):
            raise ValueError("all learn token vectors must match the data dimension")

        train_subset_size, learn_subset_size = subset_sizes
        train_size = min(train_subset_size, len(documents))
        train_indices = torch.randperm(len(documents), generator=generator)[:train_size].tolist()
        train = [documents[index] for index in train_indices]

        tokens = torch.cat(learn_documents)
        learn_size = min(learn_subset_size, len(tokens))
        learn_indices = torch.randperm(len(tokens), generator=generator)[:learn_size].to(tokens.device)
        learn_tokens = tokens[learn_indices]

        inputs, validation_inputs = learn_tokens, None
        if validation_split:
            validation_size = max(1, int(len(learn_tokens) * validation_split))
            if validation_size >= len(learn_tokens):
                raise ValueError("validation_split must leave at least one learn token for training")

            validation_inputs = learn_tokens[:validation_size]
            inputs = learn_tokens[validation_size:]

        targets = self.maxsim(train, inputs)
        self.mean = targets.mean()
        self.std = targets.std(unbiased=False)
        if not torch.isfinite(self.std) or self.std <= torch.finfo(targets.dtype).eps:
            raise ValueError("LEMUR targets have zero variance")

        targets = (targets - self.mean) / self.std
        validation_targets = None
        if validation_inputs is not None:
            validation_targets = (self.maxsim(train, validation_inputs) - self.mean) / self.std

        return {
            "dimension": dimension,
            "train_size": train_size,
            "learn": learn_tokens,
            "inputs": inputs,
            "targets": targets,
            "validation_inputs": validation_inputs,
            "validation_targets": validation_targets,
        }

    def train(
        self,
        inputs,
        targets,
        epochs,
        lr,
        batch_size,
        grad_clip,
        seed,
        validation_inputs=None,
        validation_targets=None,
    ):
        """
        Trains the MLP feature extractor and temporary readout layer.

        Args:
            inputs: learn-token vectors
            targets: standardized maxsim targets
            epochs: number of training epochs
            lr: Adam learning rate
            batch_size: training batch size
            grad_clip: optional gradient clipping value
            seed: random seed
            validation_inputs: optional held-out learn-token vectors
            validation_targets: optional standardized validation targets
        """

        inputs, targets = inputs.to(self.device), targets.to(self.device)
        if validation_inputs is not None:
            validation_inputs = validation_inputs.to(self.device)
            validation_targets = validation_targets.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        lossfn = torch.nn.MSELoss()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        best, state = float("inf"), None

        for epoch in range(epochs):
            self.model.train()
            indices = torch.randperm(len(inputs), generator=generator)
            total, batches = 0.0, 0

            for start in range(0, len(inputs), batch_size):
                batch = indices[start : start + batch_size].to(inputs.device)
                optimizer.zero_grad(set_to_none=True)
                loss = lossfn(self.model(inputs[batch]), targets[batch])
                loss.backward()

                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                optimizer.step()
                total += loss.detach().item()
                batches += 1

            score = total / batches
            metric = score
            selection_metric = "training_loss"
            if validation_inputs is not None:
                self.model.eval()
                with torch.inference_mode():
                    metric = lossfn(self.model(validation_inputs), validation_targets).item()
                selection_metric = "validation_loss"

            if metric < best:
                best = metric
                state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
                self.selected_epoch = epoch + 1
                self.selected_loss = metric
                self.selection_metric = selection_metric

        if state:
            self.model.load_state_dict(state)

    def compute_features(self, data):
        """
        Computes summed query features.

        Args:
            data: list or array of unpadded query token-vector collections

        Returns:
            fixed dimensional query features
        """

        self.check()
        documents = self.documents(data)
        self.model.eval()

        with torch.inference_mode():
            return torch.stack([self.model.features(document).sum(dim=0) / self.config["query_scale"] for document in documents])

    def compute_weights(self, data):
        """
        Computes ordinary least squares document weights.

        Args:
            data: list or array of unpadded document token-vector collections

        Returns:
            fixed dimensional document weights
        """

        self.check()
        documents = self.documents(data)

        # Precompute the feature sample SVD once per loaded encoder
        if self.pinv is None:
            self.model.eval()
            with torch.inference_mode():
                features = self.model.features(self.sample)
                left, values, right = torch.linalg.svd(features, full_matrices=False)
                tolerance = torch.finfo(values.dtype).eps * max(features.shape) * values.max()
                scales = torch.where(values > tolerance, values.reciprocal(), torch.zeros_like(values))
                self.pinv = (right.T * scales) @ left.T

        targets = (self.maxsim(documents, self.sample) - self.mean) / self.std
        return (self.pinv @ targets).T

    def documents(self, data):
        """
        Converts input data to a list of float32 tensors.

        Args:
            data: list or array of multi-vector collections

        Returns:
            list of tensors
        """

        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                data = [data]
            elif data.ndim == 3:
                data = list(data)

        documents = []
        for document in data:
            if isinstance(document, np.ndarray):
                document = torch.from_numpy(document)
            elif torch.is_tensor(document):
                document = document.detach()
            else:
                document = torch.tensor(document)

            if document.ndim != 2 or document.shape[0] == 0:
                raise ValueError("each document must be a non-empty 2D token-vector array")

            documents.append(document.to(device=self.device, dtype=torch.float32))

        return documents

    @staticmethod
    def maxsim(documents, queries):
        """
        Computes single-token maxsim targets.

        Args:
            documents: list of document token-vector tensors
            queries: token vectors used as independent single-token queries

        Returns:
            matrix with a row per query token and column per document
        """

        return torch.stack([(queries @ document.T).max(dim=1).values for document in documents], dim=1)

    def save(self, path):
        """
        Saves LEMUR artifacts.

        Args:
            path: output artifact directory
        """

        self.check()
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as output:
            json.dump(self.config, output, indent=2, sort_keys=True)
            output.write("\n")

        tensors = {name: value.detach().cpu().contiguous() for name, value in self.model.state_dict().items() if not name.startswith("output_layer.")}
        tensors.update(
            {
                "lemur.mean": self.mean.detach().cpu().reshape(1),
                "lemur.std": self.std.detach().cpu().reshape(1),
                "lemur.sample": self.sample.detach().cpu().contiguous(),
            }
        )
        safetensors.torch.save_file(tensors, os.path.join(path, "model.safetensors"))

    def load(self, path):
        """
        Loads LEMUR artifacts from a local directory or Hugging Face Hub path.

        Args:
            path: local artifact directory or Hugging Face Hub path
        """

        config = Download()(path, "config.json")
        weights = Download()(path, "model.safetensors")

        with open(config, encoding="utf-8") as source:
            self.config = json.load(source)

        self.model = LemurModel(**self.config).to(self.device)
        training = self.config.get("training", {})
        self.selected_epoch = training.get("selected_epoch")
        self.selected_loss = training.get("selected_loss")
        self.selection_metric = training.get("selection_metric")
        with safetensors.safe_open(weights, framework="pt", device=str(self.device)) as source:
            tensors = {name: source.get_tensor(name) for name in source.keys()}

        self.mean = tensors.pop("lemur.mean")[0]
        self.std = tensors.pop("lemur.std")[0]
        self.sample = tensors.pop("lemur.sample")
        self.model.load_state_dict(tensors)
        self.model.eval()
        self.pinv = None

        return self

    def check(self):
        """
        Validates that fitted or loaded LEMUR state is available.
        """

        if self.model is None or self.sample is None or self.mean is None or self.std is None:
            raise ValueError("LEMUR must be fitted or loaded before encoding")


class LemurModel(Module):
    """
    LEMUR feature extractor with an optional training-only linear readout.
    """

    # pylint: disable=R0913,R0917
    def __init__(
        self,
        model_type,
        input_dim,
        hidden_dim,
        final_hidden_dim,
        num_layers,
        activation,
        query_scale,
        seed,
        output_dim=None,
        training=None,
    ):
        """
        Creates a LEMUR feature model.

        Args:
            model_type: elm or mlp
            input_dim: token-vector dimension
            hidden_dim: MLP hidden dimension
            final_hidden_dim: fixed output dimension
            num_layers: number of MLP feature layers
            activation: feature activation
            query_scale: query feature sum scale
            seed: artifact seed
            output_dim: optional number of training target documents
            training: optional fit-selection metadata
        """

        # Unused by model construction but retained in the serialized configuration
        del query_scale, seed, training
        super().__init__()

        if model_type == "elm":
            features = [
                RandomFeatures(input_dim, final_hidden_dim, activation),
                torch.nn.LayerNorm(final_hidden_dim, elementwise_affine=False),
            ]
        elif model_type == "mlp":
            features = []
            dimensions = [input_dim] + [hidden_dim] * (num_layers - 1)
            for index, dimension in enumerate(dimensions):
                output = final_hidden_dim if index == len(dimensions) - 1 else hidden_dim
                features.extend([torch.nn.Linear(dimension, output), torch.nn.LayerNorm(output), Activation.module(activation)])
        else:
            raise ValueError("model_type must be elm or mlp")

        self.feature_extractor = torch.nn.Sequential(*features)
        self.output_layer = torch.nn.Linear(final_hidden_dim, output_dim, bias=False) if output_dim is not None else None

    def forward(self, data):
        """
        Runs feature extraction and the training readout.

        Args:
            data: input token vectors

        Returns:
            predicted standardized maxsim targets
        """

        if self.output_layer is None:
            raise ValueError("LEMUR training readout is not available in a loaded inference artifact")

        return self.output_layer(self.features(data))

    def features(self, data):
        """
        Extracts fixed dimensional token features.

        Args:
            data: input token vectors

        Returns:
            token features
        """

        return self.feature_extractor(data)


class RandomFeatures(Module):
    """
    Seeded random activation features used by the ELM model.
    """

    def __init__(self, input_dim, output_dim, activation):
        """
        Creates random activation features.

        Args:
            input_dim: input token-vector dimension
            output_dim: fixed feature dimension
            activation: feature activation
        """

        super().__init__()
        self.register_buffer("weight", torch.randn(input_dim, output_dim))
        self.activation = activation
        self.scale = math.sqrt(2.0 / output_dim)

    def forward(self, data):
        """
        Projects and activates input vectors.

        Args:
            data: input token vectors

        Returns:
            random activation features
        """

        return self.scale * Activation.function(self.activation)(data @ self.weight)


class Activation:
    """
    Activation helpers.
    """

    # Activation name to torch.nn module name. Modules and functions are resolved on demand to keep
    # this module importable when Torch isn't installed.
    MODULES = {"relu": "ReLU", "gelu": "GELU", "silu": "SiLU", "mish": "Mish"}

    @staticmethod
    def module(name):
        """
        Creates an activation module.

        Args:
            name: activation name

        Returns:
            activation module
        """

        Activation.validate(name)
        return getattr(torch.nn, Activation.MODULES[name])()

    @staticmethod
    def function(name):
        """
        Gets an activation function.

        Args:
            name: activation name

        Returns:
            activation function
        """

        Activation.validate(name)
        return getattr(torch.nn.functional, name)

    @staticmethod
    def validate(name):
        """
        Validates an activation name.

        Args:
            name: activation name
        """

        if name not in Activation.MODULES:
            raise ValueError(f"activation must be one of: {', '.join(Activation.MODULES)}")
