"""
Entity module
"""

from ..hfpipeline import HFPipeline


class Entity(HFPipeline):
    """
    Applies a token classifier to text and extracts entity/label combinations.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None):
        super().__init__("token-classification", path, quantize, gpu, model)

    def __call__(self, text, labels=None, aggregate="simple", flatten=None, join=False, workers=0):
        """
        Applies a token classifier to text and extracts entity/label combinations.

        Args:
            text: text|list
            labels: list of entity type labels to accept, defaults to None which accepts all
            aggregate: method to combine multi token entities - options are "simple" (default), "first", "average" or "max"
            flatten: flatten output to a list of labels if present. Accepts a boolean or float value to only keep scores greater than that number.
            join: joins flattened output into a string if True, ignored if flatten not set
            workers: number of concurrent workers to use for processing data, defaults to None

        Returns:
            list of (entity, entity type, score) or list of entities depending on flatten parameter
        """

        # Run token classification pipeline
        results = self.pipeline(text, aggregation_strategy=aggregate, num_workers=workers)

        # Convert results to a list if necessary
        if isinstance(text, str):
            results = [results]

        # Score threshold when flatten is set
        threshold = 0.0 if isinstance(flatten, bool) else flatten

        # Extract entities if flatten set, otherwise extract (entity, entity type, score) tuples
        outputs = []
        for result in results:
            if flatten:
                output = [r["word"] for r in result if self.accept(r["entity_group"], labels) and r["score"] >= threshold]
                outputs.append(" ".join(output) if join else output)
            else:
                outputs.append([(r["word"], r["entity_group"], float(r["score"])) for r in result if self.accept(r["entity_group"], labels)])

        return outputs[0] if isinstance(text, str) else outputs

    def accept(self, etype, labels):
        """
        Determines if entity type is in valid entity type.

        Args:
            etype: entity type
            labels: list of entities to accept

        Returns:
            if etype is accepted
        """

        return not labels or etype in labels
