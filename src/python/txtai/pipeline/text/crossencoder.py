"""
CrossEncoder module
"""

import numpy as np

from ..hfpipeline import HFPipeline


class CrossEncoder(HFPipeline):
    """
    Computes similarity between query and list of text using a cross-encoder model
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, **kwargs):
        super().__init__("text-classification", path, quantize, gpu, model, **kwargs)

    def __call__(self, query, texts, multilabel=True, workers=0):
        """
        Computes the similarity between query and list of text. Returns a list of
        (id, score) sorted by highest score, where id is the index in texts.

        This method supports query as a string or a list. If the input is a string,
        the return type is a 1D list of (id, score). If text is a list, a 2D list
        of (id, score) is returned with a row per string.

        Args:
            query: query text|list
            texts: list of text
            multilabel: labels are independent if True, scores are normalized to sum to 1 per text item if False, raw scores returned if None
            workers: number of concurrent workers to use for processing data, defaults to None

        Returns:
            list of (id, score)
        """

        scores = []
        for q in [query] if isinstance(query, str) else query:
            # Pass (query, text) pairs to model
            result = self.pipeline([{"text": q, "text_pair": t} for t in texts], top_k=None, function_to_apply="none", num_workers=workers)

            # Apply score transform function
            scores.append(self.function([r[0]["score"] for r in result], multilabel))

        # Build list of (id, score) per query sorted by highest score
        scores = [sorted(enumerate(row), key=lambda x: x[1], reverse=True) for row in scores]

        return scores[0] if isinstance(query, str) else scores

    def function(self, scores, multilabel):
        """
        Applys an output transformation function based on value of multilabel.

        Args:
            scores: input scores
            multilabel: labels are independent if True, scores are normalized to sum to 1 per text item if False, raw scores returned if None

        Returns:
            transformed scores
        """

        # Output functions
        # pylint: disable=C3001
        identity = lambda x: x
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
        softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
        function = identity if multilabel is None else sigmoid if multilabel else softmax

        # Apply output function
        return function(np.array(scores))
