"""
Similarity module
"""

import numpy as np

from .labels import Labels


class Similarity(Labels):
    """
    Computes similarity between query and list of text using a text classifier.
    """

    # pylint: disable=W0222
    def __call__(self, query, texts, multilabel=True):
        """
        Computes the similarity between query and list of text. Returns a list of
        (id, score) sorted by highest score, where id is the index in texts.

        This method supports query as a string or a list. If the input is a string,
        the return type is a 1D list of (id, score). If text is a list, a 2D list
        of (id, score) is returned with a row per string.

        Args:
            query: query text|list
            texts: list of text

        Returns:
            list of (id, score)
        """

        # Call Labels pipeline for texts using input query as the candidate label
        scores = super().__call__(texts, [query] if isinstance(query, str) else query, multilabel)

        # Sort on query index id
        scores = [[score for _, score in sorted(row)] for row in scores]

        # Transpose axes to get a list of text scores for each query
        scores = np.array(scores).T.tolist()

        # Build list of (id, score) per query sorted by highest score
        scores = [sorted(enumerate(row), key=lambda x: x[1], reverse=True) for row in scores]

        return scores[0] if isinstance(query, str) else scores
