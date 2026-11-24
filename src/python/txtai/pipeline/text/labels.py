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
        Applies a text classifier to text. Returns a list of (id, score) sorted by highest score.

        For zero shot classification (dynamic=True), a list of labels is required and id is the index of the label in labels.

        For text classification models (dynamic=False), id is the model's label id (config.id2label). A list of labels is
        optional and filters the output to those labels, otherwise all trained labels are returned. Labels can be label names
        (case-insensitive), integer label ids or label ids as strings. Labels that don't match a model label are ignored.

        This method supports text as a string or a list. If the input is a string, the return
        type is a 1D list of (id, score). If text is a list, a 2D list of (id, score) is
        returned with a row per string.

        Args:
            text: text|list
            labels: list of labels, required for zero shot classification, optional filter for text classification models
            multilabel: labels are independent if True, scores are normalized to sum to 1 per text item if False, raw scores returned if None
            flatten: returns a list of label names instead of (id, score). If True, only the top label is returned. If a number
                     (including 0.0), all labels with a score greater than or equal to that number are returned. None or False disables.
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
            flatten: flatten output to a list of labels if present, see __call__ for supported values

        Returns:
            outputs
        """

        threshold = 0.0 if isinstance(flatten, bool) else flatten

        # Resolve label filter and model label ids once for all results (fixed label models only)
        matches, ids = (None, None) if self.dynamic else (self.resolve(labels), self.ids())

        for result in results:
            if self.dynamic:
                if flatten is not None and flatten is not False:
                    result = [label for x, label in enumerate(result["labels"]) if result["scores"][x] >= threshold]
                    yield result[:1] if isinstance(flatten, bool) else result
                else:
                    yield [(labels.index(label), result["scores"][x]) for x, label in enumerate(result["labels"])]
            else:
                if flatten is not None and flatten is not False:
                    # Keep label names above threshold, filtered to requested labels if provided
                    result = [x["label"] for x in result if x["score"] >= threshold and (matches is None or ids.get(x["label"]) in matches)]
                    yield result[:1] if isinstance(flatten, bool) else result
                else:
                    # Filter results using labels, if provided
                    yield self.limit(result, labels)

    def resolve(self, labels):
        """
        Resolves a list of labels to model label ids. This is the single place label filters are resolved
        for fixed label models, used by both flattened and (id, score) outputs.

        Labels can be label names (case-insensitive), integer label ids or label ids as strings. Labels
        that don't match a model label are ignored.

        Args:
            labels: list of labels or None

        Returns:
            list of resolved label ids or None when no filter is provided
        """

        if not labels:
            return None

        # Get config
        config = self.pipeline.model.config

        # Map lowercase label names to label ids. Names from id2label take precedence, since that is
        # what the text classification pipeline outputs.
        names = {name.lower(): uid for name, uid in config.label2id.items()}
        names.update({name.lower(): uid for name, uid in self.ids().items()})

        matches = []
        for label in labels:
            # Resolve label to a label id, by id or by name. Booleans are not valid label ids.
            if isinstance(label, str):
                uid = int(label) if label.isdigit() else names.get(label.lower())
            elif isinstance(label, int) and not isinstance(label, bool):
                uid = label
            else:
                uid = None

            # Add label match, if found
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

        # Resolve label ids for model output labels
        ids = self.ids()
        result = [(ids.get(x["label"], 0), x["score"]) for x in result]

        if labels:
            matches = self.resolve(labels)

            return [(label, score) for label, score in result if label in matches]

        return result

    def ids(self):
        """
        Returns a mapping of model label name to label id. This is built from id2label, since that is what the
        text classification pipeline outputs. Some models have an id2label with label names but a label2id
        left at defaults (LABEL_0, LABEL_1...), which would otherwise map every output label to id 0.

        Returns:
            {label name: label id}
        """

        return {name: uid for uid, name in self.pipeline.model.config.id2label.items()}
