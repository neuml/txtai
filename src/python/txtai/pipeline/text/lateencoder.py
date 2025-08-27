"""
Late encoder module
"""

import numpy as np
import torch

from ...models import Models, PoolingFactory
from ..base import Pipeline


class LateEncoder(Pipeline):
    """
    Computes similarity between query and list of text using a late interaction model.
    """

    def __init__(self, path=None, **kwargs):
        # Get device
        self.device = Models.device(Models.deviceid(kwargs.get("gpu", True)))

        # Load model
        self.model = PoolingFactory.create(
            {
                "method": kwargs.get("method"),
                "path": path if path else "colbert-ir/colbertv2.0",
                "device": self.device,
                "tokenizer": kwargs.get("tokenizer"),
                "maxlength": kwargs.get("maxlength"),
                "modelargs": {**kwargs.get("vectors", {}), **{"muvera": None}},
            }
        )

    def __call__(self, query, texts, limit=None):
        """
        Computes the similarity between query and list of text. Returns a list of
        (id, score) sorted by highest score, where id is the index in texts.

        This method supports query as a string or a list. If the input is a string,
        the return type is a 1D list of (id, score). If text is a list, a 2D list
        of (id, score) is returned with a row per string.

        Args:
            query: query text|list
            texts: list of text
            limit: maximum comparisons to return, defaults to all

        Returns:
            list of (id, score)
        """

        queries = [query] if isinstance(query, str) else query

        # Encode text to vectors
        queries = self.encode(queries, "query")
        data = self.encode(texts, "data") if isinstance(texts[0], str) else texts

        # Compute maximum similarity score
        scores = []
        for q in queries:
            scores.extend(self.score(q.unsqueeze(0), data, limit))

        return scores[0] if isinstance(query, str) else scores

    def encode(self, data, category):
        """
        Encodes a batch of data using the underlying model.

        Args:
            data: input data
            category: encoding category

        Returns:
            encoded data
        """

        return torch.from_numpy(self.model.encode(data, category=category)).to(self.device)

    def score(self, queries, data, limit):
        """
        Computes the maximum similarity score between query vectors and data vectors.

        Args:
            queries: query vectors
            data: data vectors
            limit: query limit

        Returns:
            list of (id, score)
        """

        # Compute bulk dot product using einstein notation
        scores = torch.einsum("ash,bth->abst", queries, data).max(axis=-1).values.mean(axis=-1)
        scores = scores.cpu().numpy()

        # Get top n matching indices and scores
        indices = np.argpartition(-scores, limit if limit and limit < scores.shape[0] else scores.shape[0] - 1)[:, :limit]
        scores = np.take_along_axis(scores, indices, axis=1)

        results = []
        for x, index in enumerate(indices):
            results.append(list(zip(index.tolist(), scores[x].tolist())))

        return results
