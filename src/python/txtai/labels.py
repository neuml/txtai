"""
Labels module
"""

import torch

from transformers import pipeline

class Labels(object):
    """
    Applies labels to text sections using a zero shot classifier.
    """

    def __init__(self, path=None):
        """
        Creates a new Labels instance.

        Args:
            path: path to transformer model, if not provided uses a default model
        """

        self.classifier = pipeline("zero-shot-classification", model=path, tokenizer=path,
                                   device=0 if torch.cuda.is_available() else -1)

    def __call__(self, section, labels):
        """
        Applies a zero shot classifier to a text section using a list of labels.

        Args:
            section: text section
            labels: list of labels

        Returns:
            list of (label, score) for section
        """

        result = self.classifier(section, labels)
        return list(zip(result["labels"], result["scores"]))
