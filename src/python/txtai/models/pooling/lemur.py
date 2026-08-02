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
        self.selectedepoch = None
        self.selectedloss = None
        self.selectionmetric = None

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
            vectors = self.computefeatures(data)
        elif category == "data":
            vectors = self.computeweights(data)
        else:
            raise ValueError("category must be query or data")

        return vectors.detach().cpu().to(torch.float32).numpy()

    # pylint: disable=R0913,R0917
    def fit(
        self,
        data,
        output=None,
        trainsubsetsize=8192,
        learnsubsetsize=100000,
        olssamplesize=16384,
        layers=1,
        hiddendim=512,
        finalhiddendim=2048,
        activation="gelu",
        epochs=None,
        lr=3e-3,
        batchsize=512,
        gradclip=0.5,
        queryscale=32,
        seed=42,
        validationsplit=0.0,
        learn=None,
    ):
        """
        Fits a LEMUR feature encoder and OLS sample.

        Args:
            data: iterable of unpadded corpus token-vector arrays
            output: optional artifact output directory
            trainsubsetsize: maximum number of documents used to build targets
            learnsubsetsize: maximum number of token vectors used to learn features
            olssamplesize: maximum number of token vectors stored for document weights
            layers: number of MLP feature layers
            hiddendim: MLP hidden dimension
            finalhiddendim: fixed output dimension
            activation: feature activation
            epochs: required training choice; 100 is the quality MLP setting and 0 selects deterministic ELM features
            lr: Adam learning rate
            batchsize: training batch size
            gradclip: optional gradient clipping value
            queryscale: query feature sum scale
            seed: random seed
            validationsplit: fraction of sampled learn tokens held out for validation
            learn: optional iterable of learn-token arrays, defaults to data

        Returns:
            self
        """

        if epochs is None:
            raise ValueError("epochs must be set explicitly: use epochs=100 for trained MLP quality or epochs=0 for deterministic ELM features")
        if epochs < 0:
            raise ValueError("epochs must be greater than or equal to 0")
        for name, value in [
            ("olssamplesize", olssamplesize),
            ("queryscale", queryscale),
            ("layers", layers),
            ("batchsize", batchsize),
            ("trainsubsetsize", trainsubsetsize),
            ("learnsubsetsize", learnsubsetsize),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0")
        if finalhiddendim < 1:
            raise ValueError("finalhiddendim must be greater than 0")
        if validationsplit < 0 or validationsplit >= 1:
            raise ValueError("validationsplit must be greater than or equal to 0 and less than 1")

        self.selectedepoch = None
        self.selectedloss = None
        self.selectionmetric = None

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        training = self.createtrainingdata(
            data,
            learn,
            (trainsubsetsize, learnsubsetsize),
            validationsplit,
            generator,
        )

        modeltype = "elm" if epochs == 0 else "mlp"
        config = {
            "modeltype": modeltype,
            "inputdim": training["dimension"],
            "hiddendim": hiddendim,
            "finalhiddendim": finalhiddendim,
            "layers": layers,
            "activation": activation,
            "queryscale": queryscale,
            "seed": seed,
        }

        # Isolate seeded model construction from the caller's global RNG state
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = LemurModel(outputdim=training["trainsize"], **config).to(self.device)

        if epochs:
            self.train(
                training["inputs"],
                training["targets"],
                epochs,
                lr,
                batchsize,
                gradclip,
                seed,
                training["validationinputs"],
                training["validationtargets"],
            )
            config["training"] = {
                "selectedepoch": self.selectedepoch,
                "selectedloss": self.selectedloss,
                "selectionmetric": self.selectionmetric,
                "validationsplit": validationsplit,
            }

        # Persist the exact learn-token sample used by future document upserts
        learntokens = training["learn"]
        samplesize = min(olssamplesize, len(learntokens))
        sampleindices = torch.randperm(len(learntokens), generator=generator)[:samplesize].to(learntokens.device)
        self.sample = learntokens[sampleindices].detach()
        self.config = config
        self.pinv = None

        if output:
            self.save(output)

        return self

    def createtrainingdata(self, data, learn, subsetsizes, validationsplit, generator):
        """
        Creates sampled training and optional validation data.

        Args:
            data: iterable of unpadded corpus token-vector arrays
            learn: optional iterable of learn-token arrays
            subsetsizes: train-document and learn-token sample limits
            validationsplit: fraction of sampled learn tokens held out for validation
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

        learndocuments = documents if learn is None else self.documents(learn)
        if not learndocuments:
            raise ValueError("learn must contain at least one document")
        if any(document.shape[1] != dimension for document in learndocuments):
            raise ValueError("all learn token vectors must match the data dimension")

        trainsubsetsize, learnsubsetsize = subsetsizes
        trainsize = min(trainsubsetsize, len(documents))
        trainindices = torch.randperm(len(documents), generator=generator)[:trainsize].tolist()
        train = [documents[index] for index in trainindices]

        tokens = torch.cat(learndocuments)
        learnsize = min(learnsubsetsize, len(tokens))
        learnindices = torch.randperm(len(tokens), generator=generator)[:learnsize].to(tokens.device)
        learntokens = tokens[learnindices]

        inputs, validationinputs = learntokens, None
        if validationsplit:
            validationsize = max(1, int(len(learntokens) * validationsplit))
            if validationsize >= len(learntokens):
                raise ValueError("validationsplit must leave at least one learn token for training")

            validationinputs = learntokens[:validationsize]
            inputs = learntokens[validationsize:]

        targets = self.maxsim(train, inputs)
        self.mean = targets.mean()
        self.std = targets.std(unbiased=False)
        if not torch.isfinite(self.std) or self.std <= torch.finfo(targets.dtype).eps:
            raise ValueError("LEMUR targets have zero variance")

        targets = (targets - self.mean) / self.std
        validationtargets = None
        if validationinputs is not None:
            validationtargets = (self.maxsim(train, validationinputs) - self.mean) / self.std

        return {
            "dimension": dimension,
            "trainsize": trainsize,
            "learn": learntokens,
            "inputs": inputs,
            "targets": targets,
            "validationinputs": validationinputs,
            "validationtargets": validationtargets,
        }

    def train(
        self,
        inputs,
        targets,
        epochs,
        lr,
        batchsize,
        gradclip,
        seed,
        validationinputs=None,
        validationtargets=None,
    ):
        """
        Trains the MLP feature extractor and temporary readout layer.

        Args:
            inputs: learn-token vectors
            targets: standardized maxsim targets
            epochs: number of training epochs
            lr: Adam learning rate
            batchsize: training batch size
            gradclip: optional gradient clipping value
            seed: random seed
            validationinputs: optional held-out learn-token vectors
            validationtargets: optional standardized validation targets
        """

        inputs, targets = inputs.to(self.device), targets.to(self.device)
        if validationinputs is not None:
            validationinputs = validationinputs.to(self.device)
            validationtargets = validationtargets.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        lossfn = torch.nn.MSELoss()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        best, state = float("inf"), None

        for epoch in range(epochs):
            self.model.train()
            indices = torch.randperm(len(inputs), generator=generator)
            total, batches = 0.0, 0

            for start in range(0, len(inputs), batchsize):
                batch = indices[start : start + batchsize].to(inputs.device)
                optimizer.zero_grad(set_to_none=True)
                loss = lossfn(self.model(inputs[batch]), targets[batch])
                loss.backward()

                if gradclip is not None and gradclip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradclip)

                optimizer.step()
                total += loss.detach().item()
                batches += 1

            score = total / batches
            metric = score
            selectionmetric = "trainingloss"
            if validationinputs is not None:
                self.model.eval()
                with torch.inference_mode():
                    metric = lossfn(self.model(validationinputs), validationtargets).item()
                selectionmetric = "validationloss"

            if metric < best:
                best = metric
                state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
                self.selectedepoch = epoch + 1
                self.selectedloss = metric
                self.selectionmetric = selectionmetric

        if state:
            self.model.load_state_dict(state)

    def computefeatures(self, data):
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
            return torch.stack([self.model.features(document).sum(dim=0) / self.config["queryscale"] for document in documents])

    def computeweights(self, data):
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

        tensors = {name: value.detach().cpu().contiguous() for name, value in self.model.state_dict().items() if not name.startswith("outputlayer.")}
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
        self.selectedepoch = training.get("selectedepoch")
        self.selectedloss = training.get("selectedloss")
        self.selectionmetric = training.get("selectionmetric")
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
        modeltype,
        inputdim,
        hiddendim,
        finalhiddendim,
        layers,
        activation,
        queryscale,
        seed,
        outputdim=None,
        training=None,
    ):
        """
        Creates a LEMUR feature model.

        Args:
            modeltype: elm or mlp
            inputdim: token-vector dimension
            hiddendim: MLP hidden dimension
            finalhiddendim: fixed output dimension
            layers: number of MLP feature layers
            activation: feature activation
            queryscale: query feature sum scale
            seed: artifact seed
            outputdim: optional number of training target documents
            training: optional fit-selection metadata
        """

        # Unused by model construction but retained in the serialized configuration
        del queryscale, seed, training
        super().__init__()

        if modeltype == "elm":
            features = [
                RandomFeatures(inputdim, finalhiddendim, activation),
                torch.nn.LayerNorm(finalhiddendim, elementwise_affine=False),
            ]
        elif modeltype == "mlp":
            features = []
            dimensions = [inputdim] + [hiddendim] * (layers - 1)
            for index, dimension in enumerate(dimensions):
                output = finalhiddendim if index == len(dimensions) - 1 else hiddendim
                features.extend([torch.nn.Linear(dimension, output), torch.nn.LayerNorm(output), Activation.module(activation)])
        else:
            raise ValueError("modeltype must be elm or mlp")

        self.featureextractor = torch.nn.Sequential(*features)
        self.outputlayer = torch.nn.Linear(finalhiddendim, outputdim, bias=False) if outputdim is not None else None

    def forward(self, data):
        """
        Runs feature extraction and the training readout.

        Args:
            data: input token vectors

        Returns:
            predicted standardized maxsim targets
        """

        if self.outputlayer is None:
            raise ValueError("LEMUR training readout is not available in a loaded inference artifact")

        return self.outputlayer(self.features(data))

    def features(self, data):
        """
        Extracts fixed dimensional token features.

        Args:
            data: input token vectors

        Returns:
            token features
        """

        return self.featureextractor(data)


class RandomFeatures(Module):
    """
    Seeded random activation features used by the ELM model.
    """

    def __init__(self, inputdim, outputdim, activation):
        """
        Creates random activation features.

        Args:
            inputdim: input token-vector dimension
            outputdim: fixed feature dimension
            activation: feature activation
        """

        super().__init__()
        self.register_buffer("weight", torch.randn(inputdim, outputdim))
        self.activation = activation
        self.scale = math.sqrt(2.0 / outputdim)

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
