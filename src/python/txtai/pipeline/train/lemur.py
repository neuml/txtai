"""
LEMUR trainer module
"""

import random
import sys

from ...models import Lemur, Models, PoolingFactory
from ...models.pooling.lemur import LemurModel
from ...util import Library
from ..base import Pipeline


library = Library()
np = library.numpy()
torch = library.torch()
tqdm = library.tqdm()


class LemurTrainer(Pipeline):
    """
    Trains LEMUR artifacts for a late interaction model and corpus.
    """

    # pylint: disable=R0913,R0917
    def __call__(
        self,
        path,
        data,
        output,
        gpu=True,
        method=None,
        tokenizer=None,
        maxlength=None,
        vectors=None,
        learn=None,
        learncategory="query",
        corpussubsetsize=None,
        validationsplit=0.0,
        **kwargs,
    ):
        """
        Trains a LEMUR feature encoder.

        Args:
            path: late interaction model path
            data: iterable of corpus texts
            output: artifact output directory
            gpu: tensor accelerator setting
            method: optional pooling method
            tokenizer: optional tokenizer path
            maxlength: maximum token length
            vectors: additional model arguments
            learn: optional iterable of texts used to learn features, defaults to data
            learncategory: encoder category for learn texts (data or query)
            corpussubsetsize: optional maximum number of corpus texts selected before encoding
            validationsplit: fraction of sampled learn tokens held out for validation
            kwargs: LEMUR fit arguments

        Returns:
            fitted LEMUR encoder
        """

        data = list(data)
        if not data:
            raise ValueError("data must contain at least one corpus text")
        if learncategory not in ("data", "query"):
            raise ValueError("learncategory must be data or query")
        if kwargs.get("epochs") is None:
            raise ValueError("epochs must be set explicitly: use epochs=100 for trained MLP quality or epochs=0 for deterministic ELM features")
        if corpussubsetsize is not None:
            if isinstance(corpussubsetsize, bool) or not isinstance(corpussubsetsize, int) or corpussubsetsize <= 0:
                raise ValueError("corpussubsetsize must be a positive integer")
            if corpussubsetsize < len(data):
                indices = sorted(random.Random(kwargs.get("seed", 42)).sample(range(len(data)), corpussubsetsize))
                data = [data[index] for index in indices]

        learn = list(learn) if learn is not None else None
        if learn is not None and not learn:
            raise ValueError("learn must contain at least one text")

        deviceid = Models.deviceid(gpu)
        modelargs = {**(vectors if vectors else {}), **{"muvera": None, "lemur": None}}
        centerconfigured = vectors is not None and "center" in vectors
        if not centerconfigured:
            modelargs["center"] = False
        pooling = PoolingFactory.create(
            {
                "method": method,
                "path": path,
                "device": deviceid,
                "tokenizer": tokenizer,
                "maxlength": maxlength,
                "modelargs": modelargs,
            }
        )

        # A batch size of one preserves each document's true token count. The late
        # pooling path normalizes token rows before returning raw multi-vectors.
        documents = [pooling.encode([text], batch=1, category="data")[0] for text in data]
        learndocuments = None
        if learn is not None or learncategory != "data":
            learn = data if learn is None else learn
            learndocuments = [pooling.encode([text], batch=1, category=learncategory)[0] for text in learn]

        centermean = None
        if not centerconfigured:
            centermean = np.concatenate(documents).mean(axis=0)
            pooling.center = {"scope": "collection", "mean": centermean}
            documents = pooling.centerdata(documents)
            if learndocuments is not None:
                learndocuments = pooling.centerdata(learndocuments)

        return self.fit(
            documents,
            output=output,
            device=Models.device(deviceid),
            validationsplit=validationsplit,
            learn=learndocuments,
            centermean=centermean,
            **kwargs,
        )

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
        device=None,
        centermean=None,
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
            device: tensor device
            centermean: optional collection token mean stored with the artifact

        Returns:
            fitted LEMUR encoder
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

        lemur = Lemur(device=device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        training = self.createtrainingdata(
            lemur,
            data,
            learn,
            (trainsubsetsize, learnsubsetsize),
            validationsplit,
            generator,
        )

        if centermean is not None:
            centermean = torch.as_tensor(centermean, dtype=torch.float32, device=lemur.device)
            if centermean.ndim != 1 or centermean.shape[0] != training["dimension"] or not torch.isfinite(centermean).all():
                raise ValueError("centermean must be a finite one-dimensional array matching the token dimension")
            lemur.center = centermean.detach()

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
            lemur.model = LemurModel(outputdim=training["trainsize"], **config).to(lemur.device)

        if epochs:
            self.train(
                lemur,
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
                "selectedepoch": lemur.selectedepoch,
                "selectedloss": lemur.selectedloss,
                "selectionmetric": lemur.selectionmetric,
                "validationsplit": validationsplit,
            }

        # Persist the exact learn-token sample used by future document upserts
        learntokens = training["learn"]
        samplesize = min(olssamplesize, len(learntokens))
        sampleindices = torch.randperm(len(learntokens), generator=generator)[:samplesize].to(learntokens.device)
        lemur.sample = learntokens[sampleindices].detach()
        lemur.config = config
        lemur.pinv = None

        if output:
            lemur.save(output)

        return lemur

    def createtrainingdata(self, lemur, data, learn, subsetsizes, validationsplit, generator):
        """
        Creates sampled training and optional validation data.

        Args:
            lemur: LEMUR encoder receiving fitted state
            data: iterable of unpadded corpus token-vector arrays
            learn: optional iterable of learn-token arrays
            subsetsizes: train-document and learn-token sample limits
            validationsplit: fraction of sampled learn tokens held out for validation
            generator: seeded random generator

        Returns:
            sampled training data
        """

        documents = lemur.documents(data)
        if not documents:
            raise ValueError("data must contain at least one document")

        dimension = documents[0].shape[1]
        if any(document.shape[1] != dimension for document in documents):
            raise ValueError("all token vectors must have the same dimension")

        learndocuments = documents if learn is None else lemur.documents(learn)
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

        targets = lemur.maxsim(train, inputs)
        lemur.mean = targets.mean()
        lemur.std = targets.std(unbiased=False)
        if not torch.isfinite(lemur.std) or lemur.std <= torch.finfo(targets.dtype).eps:
            raise ValueError("LEMUR targets have zero variance")

        targets = (targets - lemur.mean) / lemur.std
        validationtargets = None
        if validationinputs is not None:
            validationtargets = (lemur.maxsim(train, validationinputs) - lemur.mean) / lemur.std

        return {
            "dimension": dimension,
            "trainsize": trainsize,
            "learn": learntokens,
            "inputs": inputs,
            "targets": targets,
            "validationinputs": validationinputs,
            "validationtargets": validationtargets,
        }

    # pylint: disable=R0913,R0917
    def train(
        self,
        lemur,
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
            lemur: LEMUR encoder receiving fitted state
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

        inputs, targets = inputs.to(lemur.device), targets.to(lemur.device)
        if validationinputs is not None:
            validationinputs = validationinputs.to(lemur.device)
            validationtargets = validationtargets.to(lemur.device)

        optimizer = torch.optim.Adam(lemur.model.parameters(), lr=lr)
        lossfn = torch.nn.MSELoss()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        best, state = float("inf"), None

        progress = tqdm.tqdm(range(epochs), desc="LEMUR training", unit="epoch", disable=not sys.stderr.isatty())
        for epoch in progress:
            lemur.model.train()
            indices = torch.randperm(len(inputs), generator=generator)
            total, batches = 0.0, 0

            for start in range(0, len(inputs), batchsize):
                batch = indices[start : start + batchsize].to(inputs.device)
                optimizer.zero_grad(set_to_none=True)
                loss = lossfn(lemur.model(inputs[batch]), targets[batch])
                loss.backward()

                if gradclip is not None and gradclip > 0:
                    torch.nn.utils.clip_grad_norm_(lemur.model.parameters(), gradclip)

                optimizer.step()
                total += loss.detach().item()
                batches += 1

            score = total / batches
            metric = score
            selectionmetric = "trainingloss"
            if validationinputs is not None:
                lemur.model.eval()
                with torch.inference_mode():
                    metric = lossfn(lemur.model(validationinputs), validationtargets).item()
                selectionmetric = "validationloss"

            progress.set_postfix({selectionmetric.replace("loss", " loss"): f"{metric:.6f}"})
            if metric < best:
                best = metric
                state = {name: value.detach().cpu().clone() for name, value in lemur.model.state_dict().items()}
                lemur.selectedepoch = epoch + 1
                lemur.selectedloss = metric
                lemur.selectionmetric = selectionmetric

        if state:
            lemur.model.load_state_dict(state)
