"""
Similarity module
"""

import numpy as np

from .crossencoder import CrossEncoder
from .labels import Labels


class Similarity(Labels):
    """
    Computes similarity between query and list of text using a text classifier.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, dynamic=True, crossencode=False, **kwargs):
        # Use zero-shot classification if dynamic is True and crossencode is False, otherwise use standard text classification
        super().__init__(path, quantize, gpu, model, False if crossencode else dynamic, **kwargs)

        # Load as a cross-encoder if crossencode set to True
        self.crossencoder = CrossEncoder(model=self.pipeline) if crossencode else None

    # pylint: disable=W0222
    def __call__(self, query, texts, multilabel=True, **kwargs):
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
            kwargs: additional keyword args

        Returns:
            list of (id, score)
        """

        if self.crossencoder:
            # pylint: disable=E1102
            return self.crossencoder(query, texts, multilabel)

        # Call Labels pipeline for texts using input query as the candidate label
        scores = super().__call__(texts, [query] if isinstance(query, str) else query, multilabel, **kwargs)

        # Sort on query index id
        scores = [[score for _, score in sorted(row)] for row in scores]

        # Transpose axes to get a list of text scores for each query
        scores = np.array(scores).T.tolist()

        # Build list of (id, score) per query sorted by highest score
        scores = [sorted(enumerate(row), key=lambda x: x[1], reverse=True) for row in scores]

        return scores[0] if isinstance(query, str) else scores
