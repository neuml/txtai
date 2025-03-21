"""
Entity module
"""

# Conditional import
try:
    from gliner import GLiNER

    GLINER = True
except ImportError:
    GLINER = False

from huggingface_hub.errors import HFValidationError
from transformers.utils import cached_file

from ...models import Models
from ..hfpipeline import HFPipeline


class Entity(HFPipeline):
    """
    Applies a token classifier to text and extracts entity/label combinations.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, **kwargs):
        # Create a new entity pipeline
        self.gliner = self.isgliner(path)
        if self.gliner:
            if not GLINER:
                raise ImportError('GLiNER is not available - install "pipeline" extra to enable')

            # GLiNER entity pipeline
            self.pipeline = GLiNER.from_pretrained(path)
            self.pipeline = self.pipeline.to(Models.device(Models.deviceid(gpu)))
        else:
            # Standard entity pipeline
            super().__init__("token-classification", path, quantize, gpu, model, **kwargs)

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
        results = self.execute(text, labels, aggregate, workers)

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

    def isgliner(self, path):
        """
        Tests if path is a GLiNER model.

        Args:
            path: model path

        Returns:
            True if this is a GLiNER model, False otherwise
        """

        try:
            # Test if this model has a gliner_config.json file
            return cached_file(path_or_repo_id=path, filename="gliner_config.json") is not None

        # Ignore this error - invalid repo or directory
        except (HFValidationError, OSError):
            pass

        return False

    def execute(self, text, labels, aggregate, workers):
        """
        Runs the entity extraction pipeline.

        Args:
            text: text|list
            labels: list of entity type labels to accept, defaults to None which accepts all
            aggregate: method to combine multi token entities - options are "simple" (default), "first", "average" or "max"
            workers: number of concurrent workers to use for processing data, defaults to None

        Returns:
            list of entities and labels
        """

        if self.gliner:
            # Extract entities with GLiNER. Use default CoNLL-2003 labels when not otherwise provided.
            results = self.pipeline.batch_predict_entities(
                text if isinstance(text, list) else [text], labels if labels else ["person", "organization", "location"]
            )

            # Map results to same format as Transformers token classifier
            entities = []
            for result in results:
                entities.append([{"word": x["text"], "entity_group": x["label"], "score": x["score"]} for x in result])

            # Return extracted entities
            return entities if isinstance(text, list) else entities[0]

        # Standard Transformers token classification pipeline
        return self.pipeline(text, aggregation_strategy=aggregate, num_workers=workers)

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
