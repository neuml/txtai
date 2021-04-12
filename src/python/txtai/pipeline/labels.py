"""
Labels module
"""

from .hfpipeline import HFPipeline


class Labels(HFPipeline):
    """
    Applies a zero shot classifier to text using a list of labels.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None):
        super().__init__("zero-shot-classification", path, quantize, gpu, model)

    def __call__(self, text, labels, multilabel=False):
        """
        Applies a zero shot classifier to text using a list of labels. Returns a list of
        (id, score) sorted by highest score, where id is the index in labels.

        This method supports text as a string or a list. If the input is a string, the return
        type is a 1D list of (id, score). If text is a list, a 2D list of (id, score) is
        returned with a row per string.

        Args:
            text: text|list
            labels: list of labels

        Returns:
            list of (id, score)
        """

        # Run ZSL pipeline
        results = self.pipeline(text, labels, multi_label=multilabel, truncation=True)

        # Convert results to a list if necessary
        if not isinstance(results, list):
            results = [results]

        # Build list of (id, score)
        scores = []
        for result in results:
            scores.append([(labels.index(label), result["scores"][x]) for x, label in enumerate(result["labels"])])

        return scores[0] if isinstance(text, str) else scores
