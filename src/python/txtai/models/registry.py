"""
Registry module
"""

from collections import namedtuple

from transformers.models.auto.modeling_auto import (
    MODEL_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
)
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING


class Registry:
    """
    Methods to register models and fully support pipelines.
    """

    @staticmethod
    def register(model):
        """
        Registers a model with auto model and tokenizer configuration to fully support pipelines.

        Args:
            model: model to register
        """

        # Add references for this class to supported AutoModel classes
        name = model.__class__.__name__
        if name not in MODEL_MAPPING:
            for mapping in [MODEL_MAPPING, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, MODEL_FOR_QUESTION_ANSWERING_MAPPING]:
                Registry.autoadd(mapping, name, model.__class__)

        # Add references for this class to support pipeline AutoTokenizers
        if hasattr(model, "config") and type(model.config) not in TOKENIZER_MAPPING:
            Registry.autoadd(TOKENIZER_MAPPING, type(model.config), type(model.config).__name__)

    @staticmethod
    def autoadd(mapping, key, value):
        """
        Adds auto model configuration to fully support pipelines.

        Args:
            mapping: auto model mapping
            key: key to add
            value: value to add
        """

        # pylint: disable=W0212
        Params = namedtuple("Params", ["config", "model"])
        params = Params(key, value)

        mapping._modules[key] = params
        mapping._config_mapping[key] = "config"
        mapping._reverse_config_mapping[value] = key
        mapping._model_mapping[key] = "model"
