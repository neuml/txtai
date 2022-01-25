"""
Entity module
"""

from ..hfpipeline import HFPipeline


class Entity(HFPipeline):
    """
    Applies a token classifier to text. Extracts known entity/label combinations.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None):
        super().__init__("token-classification", path, quantize, gpu, model)

    def __call__(self, text, aggregate="simple", workers=0):
        """
        Applies a token classifier to text.

        Args:
            text: text|list
            aggregate: method to combine multi token entities - options are "simple" (default), "first", "average" or "max"
            workers: number of concurrent workers to use for processing data, defaults to None

        Returns:
            list of (entity, entity type, score)
        """

        # Run token classification pipeline
        results = self.pipeline(text, aggregation_strategy=aggregate, num_workers=workers)

        # Convert results to a list if necessary
        if isinstance(text, str):
            results = [results]

        # Extract (entity, entity type, score) tuples
        outputs = []
        for result in results:
            outputs.append([(r["word"], r["entity_group"], float(r["score"])) for r in result])

        return outputs[0] if isinstance(text, str) else outputs
