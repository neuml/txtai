"""
Registry module
"""

# Core library imports
from ..util import Library

transformers = Library().transformers()


class Registry:
    """
    Methods to register models and fully support pipelines.
    """

    @staticmethod
    def register(model, config=None):
        """
        Registers a model with auto model and tokenizer configuration to fully support pipelines.

        Args:
            model: model to register
            config: config class name
        """

        # Default config class to model class if not provided
        config = config if config else model.__class__

        # Default model config_class if empty
        if hasattr(model.__class__, "config_class") and not model.__class__.config_class:
            model.__class__.config_class = config

        # Add references for this class to supported AutoModel classes
        for mapping in [transformers.AutoModel, transformers.AutoModelForQuestionAnswering, transformers.AutoModelForSequenceClassification]:
            mapping.register(config, model.__class__)

        # Add references for this class to support pipeline AutoTokenizers
        mappings = transformers.models.auto.tokenization_auto.TOKENIZER_MAPPING
        if hasattr(model, "config") and type(model.config) not in mappings:
            mappings.register(type(model.config), type(model.config).__name__)
