"""
LEMUR trainer module
"""

import random

from ...models import Lemur, Models, PoolingFactory
from ..base import Pipeline


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

        lemur = Lemur(device=Models.device(deviceid))
        return lemur.fit(documents, output=output, validationsplit=validationsplit, learn=learndocuments, **kwargs)
