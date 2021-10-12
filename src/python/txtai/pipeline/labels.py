"""
Labels module
"""

from .hfpipeline import HFPipeline


class Labels(HFPipeline):
    """
    Applies a text classifier to text. Supports zero shot and standard text classification models
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, dynamic=True):
        super().__init__("zero-shot-classification" if dynamic else "text-classification", path, quantize, gpu, model)

        # Set if labels are dynamic (zero shot) or fixed (standard text classification)
        self.dynamic = dynamic

    def __call__(self, text, labels=None, multilabel=False, workers=0):
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
            multilabel: labels are independent if True, scores are normalized to sum to 1 per text item if False, raw scores returned if None
            workers: number of parallel workers to use for processing data, defaults to None

        Returns:
            list of (id, score)
        """

        if self.dynamic:
            # Run zero shot classification pipeline
            results = self.pipeline(text, labels, multi_label=multilabel, truncation=True, num_workers=workers)
        else:
            # Set classification function based on inputs
            function = "none" if multilabel is None else "sigmoid" if multilabel or len(self.labels()) == 1 else "softmax"

            # Run text classification pipeline
            results = self.pipeline(text, return_all_scores=True, function_to_apply=function, num_workers=workers)

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

        # Extract scores from result
        result = [x["score"] for x in result]

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
