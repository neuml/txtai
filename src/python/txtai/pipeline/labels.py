"""
Labels module
"""

import numpy as np

from .hfpipeline import HFPipeline


class Labels(HFPipeline):
    """
    Applies a text classifier to text. Supports zero shot and standard text classification models
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, dynamic=True):
        super().__init__("zero-shot-classification" if dynamic else "text-classification", path, quantize, gpu, model)

        # Set if labels are dynamic (zero shot) or fixed (standard text classification)
        self.dynamic = dynamic

    def __call__(self, text, labels=None, multilabel=False):
        """
        Applies a text classifier to text. Returns a list of (id, score) sorted by highest score,
        where id is the index in labels. For zero shot classification, a list of labels is required.
        For text classification models, a list of labels is optional, otherwise all trained labels are returned.

        This method supports text as a string or a list. If the input is a string, the return
        type is a 1D list of (id, score). If text is a list, a 2D list of (id, score) is
        returned with a row per string.

        Args:
            text: text|list
            labels: list of labels
            multilabel: labels are independent if True, otherwise scores are normalized to sum to 1 per text item

        Returns:
            list of (id, score)
        """

        if self.dynamic:
            # Run zero shot classification pipeline
            results = self.pipeline(text, labels, multi_label=multilabel, truncation=True)
        else:
            # Run text classification pipeline
            results = self.textclassify(text, multilabel)

        # Convert results to a list if necessary
        if not isinstance(results, list):
            results = [results]

        # Build list of (id, score)
        scores = []
        for result in results:
            if self.dynamic:
                scores.append([(labels.index(label), result["scores"][x]) for x, label in enumerate(result["labels"])])
            else:
                # Filter results using labels, if provided
                result = self.limit(result, labels)

                scores.append(sorted(enumerate(result), key=lambda x: x[1], reverse=True))

        return scores[0] if isinstance(text, str) else scores

    def textclassify(self, text, multilabel):
        """
        Wrapper of the text-classification pipeline to be more flexible with the logits transformation
        function. While these methods are in later versions of Hugging Face Transformers, this method supports
        older versions.

        Args:
            text: text|list
            multilabel: labels are independent if True, otherwise scores are normalized to sum to 1 per text item

        Returns:
            list of scores for each label in model
        """

        # pylint: disable=W0212
        inputs = self.pipeline._parse_and_tokenize(text, truncation=True)
        outputs = self.pipeline._forward(inputs)

        # Apply sigmoid to outputs
        if multilabel or len(self.labels()) == 1:
            return (1.0 / (1.0 + np.exp(-outputs))).tolist()

        # Apply softmax to outputs
        maxes = np.max(outputs, axis=-1, keepdims=True)
        shift = np.exp(outputs - maxes)
        return (shift / shift.sum(axis=-1, keepdims=True)).tolist()

    def labels(self):
        """
        Returns a list of all text classification model labels sorted in index order.

        Returns:
            list of labels
        """

        return list(self.pipeline.model.config.id2label.values())

    def limit(self, result, labels):
        """
        Filter result using labels. If labels is None, original result is returned.

        Args:
            result: results array
            labels: list of labels or None

        Returns:
            filtered results
        """

        if labels:
            config = self.pipeline.model.config
            indices = []
            for label in labels:
                # Lookup label keys from model config
                if label.isdigit():
                    label = int(label)
                    keys = list(config.id2label.keys())
                else:
                    label = label.lower()
                    keys = [x.lower() for x in config.label2id.keys()]

                # Get index, default to 0 if not found
                indices.append(keys.index(label) if label in keys else 0)

            return [result[i] for i in indices]

        return result
