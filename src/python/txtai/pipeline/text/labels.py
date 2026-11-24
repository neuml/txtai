"""
Labels module
"""

from ..hfpipeline import HFPipeline


class Labels(HFPipeline):
    """
    Applies a text classifier to text. Supports zero shot and standard text classification models
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, dynamic=True, **kwargs):
        super().__init__("zero-shot-classification" if dynamic else "text-classification", path, quantize, gpu, model, **kwargs)

        # Set if labels are dynamic (zero shot) or fixed (standard text classification)
        self.dynamic = dynamic

    def __call__(self, text, labels=None, multilabel=False, flatten=None, workers=0, **kwargs):
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
            flatten: flatten output to a list of labels if present. Accepts a boolean or float value to only keep scores greater than that number.
            workers: number of concurrent workers to use for processing data, defaults to 0
            kwargs: additional keyword args

        Returns:
            list of (id, score) or list of labels depending on flatten parameter
        """

        if self.dynamic:
            # Run zero shot classification pipeline
            results = self.pipeline(text, labels, multi_label=multilabel, truncation=True, num_workers=workers, **kwargs)
        else:
            # Set classification function based on inputs
            function = "none" if multilabel is None else "sigmoid" if multilabel or len(self.labels()) == 1 else "softmax"

            # Run text classification pipeline
            results = self.pipeline(text, top_k=None, function_to_apply=function, num_workers=workers, **kwargs)

        # Convert results to a list if necessary
        if isinstance(text, str):
            results = [results]

        # Build list of outputs and return
        outputs = self.outputs(results, labels, flatten)

        # Stream outputs into list, if necessary
        outputs = list(outputs) if isinstance(results, list) else outputs

        # Return format that matches input format
        return outputs[0] if isinstance(text, str) else outputs

    def labels(self):
        """
        Returns a list of all text classification model labels sorted in index order.

        Returns:
            list of labels
        """

        return list(self.pipeline.model.config.id2label.values())

    def outputs(self, results, labels, flatten):
        """
        Processes pipeline results and builds outputs.

        Args:
            results: pipeline results
            labels: list of labels
            flatten: flatten output to a list of labels if present. Accepts a boolean or float value to only keep scores greater than that number.

        Returns:
            outputs
        """

        threshold = 0.0 if isinstance(flatten, bool) else flatten

        for result in results:
            if self.dynamic:
                if flatten is not None and flatten is not False:
                    result = [label for x, label in enumerate(result["labels"]) if result["scores"][x] >= threshold]
                    yield result[:1] if isinstance(flatten, bool) else result
                else:
                    yield [(labels.index(label), result["scores"][x]) for x, label in enumerate(result["labels"])]
            else:
                if flatten is not None and flatten is not False:
                    matches = self.resolve(labels)
                    config = self.pipeline.model.config
                    result = [
                        x["label"]
                        for x in result
                        if x["score"] >= threshold
                        and (matches is None or config.label2id.get(x["label"]) in matches)
                    ]
                    yield result[:1] if isinstance(flatten, bool) else result
                else:
                    # Filter results using labels, if provided
                    yield self.limit(result, labels)

    def resolve(self, labels):
        """
        Resolve label names and IDs into valid model label IDs.

        Args:
            labels: list of labels or None

        Returns:
            list of resolved label IDs or None when no filter is provided
        """

        if not labels:
            return None

        config = self.pipeline.model.config
        names = {name.lower(): uid for name, uid in config.label2id.items()}
        names.update({name.lower(): uid for uid, name in config.id2label.items()})

        matches = []
        for label in labels:
            if isinstance(label, str):
                uid = int(label) if label.isdigit() else names.get(label.lower())
            elif isinstance(label, int) and not isinstance(label, bool):
                uid = label
            else:
                uid = None

            if uid in config.id2label and uid not in matches:
                matches.append(uid)

        return matches

    def limit(self, result, labels):
        """
        Filter result using labels. If labels is None, original result is returned.

        Args:
            result: results array sorted by score descending
            labels: list of labels or None

        Returns:
            filtered results
        """

        # Get config
        config = self.pipeline.model.config

        # Resolve label ids for labels
        result = [(config.label2id.get(x["label"], 0), x["score"]) for x in result]

        if labels:
            matches = self.resolve(labels)

            return [(label, score) for label, score in result if label in matches]

        return result
