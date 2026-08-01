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
        learn_category="query",
        corpus_subset_size=None,
        validation_split=0.0,
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
            learn_category: encoder category for learn texts (data or query)
            corpus_subset_size: optional maximum number of corpus texts selected before encoding
            validation_split: fraction of sampled learn tokens held out for validation
            kwargs: LEMUR fit arguments

        Returns:
            fitted LEMUR encoder
        """

        data = list(data)
        if not data:
            raise ValueError("data must contain at least one corpus text")
        if learn_category not in ("data", "query"):
            raise ValueError("learn_category must be data or query")
        if kwargs.get("epochs") is None:
            raise ValueError("epochs must be set explicitly: use epochs=100 for trained MLP quality or epochs=0 for deterministic ELM features")
        if corpus_subset_size is not None:
            if isinstance(corpus_subset_size, bool) or not isinstance(corpus_subset_size, int) or corpus_subset_size <= 0:
                raise ValueError("corpus_subset_size must be a positive integer")
            if corpus_subset_size < len(data):
                indices = sorted(random.Random(kwargs.get("seed", 42)).sample(range(len(data)), corpus_subset_size))
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
        learn_documents = None
        if learn is not None or learn_category != "data":
            learn = data if learn is None else learn
            learn_documents = [pooling.encode([text], batch=1, category=learn_category)[0] for text in learn]

        lemur = Lemur(device=Models.device(deviceid))
        return lemur.fit(documents, output=output, validation_split=validation_split, learn=learn_documents, **kwargs)
