"""
ONNX module
"""

from collections import namedtuple

# Conditional import
try:
    from onnxruntime import InferenceSession, SessionOptions

    ONNX_RUNTIME = True
except ImportError:
    ONNX_RUNTIME = False

import numpy as np
import torch

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.modeling_auto import (
    MODEL_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
)
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from transformers.modeling_utils import PreTrainedModel

# pylint: disable=W0223
class OnnxModel(PreTrainedModel):
    """
    Provides a Transformers/PyTorch compatible interface for ONNX models. Handles casting inputs
    and outputs with minimal to no copying of data.
    """

    def __init__(self, model):
        """
        Creates a new OnnxModel.

        Args:
            model: path to model or InferenceSession
        """

        if not ONNX_RUNTIME:
            raise ImportError('onnxruntime is not available - install "model" extra to enable')

        super().__init__(OnnxConfig())

        # Create ONNX session
        self.model = InferenceSession(model, SessionOptions())

        # Add references for this class to supported AutoModel classes
        name = self.__class__.__name__
        if name not in MODEL_MAPPING:
            for mapping in [MODEL_MAPPING, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, MODEL_FOR_QUESTION_ANSWERING_MAPPING]:
                self.autoadd(mapping, name, self.__class__)

        # Add references for this class to support pipeline AutoTokenizers
        if type(self.config) not in TOKENIZER_MAPPING:
            self.autoadd(TOKENIZER_MAPPING, type(self.config), type(self.config).__name__)

    def autoadd(self, mapping, key, value):
        """
        Adds OnnxModel to auto model configuration to fully support pipelines.

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

    def forward(self, **inputs):
        """
        Runs inputs through an ONNX model and returns outputs. This method handles casting inputs
        and outputs between torch tensors and numpy arrays as shared memory (no copy).

        Args:
            inputs: model inputs

        Returns:
            model outputs
        """

        inputs = self.parse(inputs)

        # Run inputs through ONNX model
        results = self.model.run(None, inputs)

        # pylint: disable=E1101
        return torch.from_numpy(np.array(results))

    def parse(self, inputs):
        """
        Parse model inputs and handle converting to ONNX compatible inputs.

        Args:
            inputs: model inputs

        Returns:
            ONNX compatible model inputs
        """

        features = {}

        # Select features from inputs
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in inputs:
                value = inputs[key]

                # Cast torch tensors to numpy
                if hasattr(value, "cpu"):
                    value = value.cpu().numpy()

                # Cast to numpy array if not already one
                features[key] = np.asarray(value)

        return features


class OnnxConfig(PretrainedConfig):
    """
    Configuration for ONNX models.
    """
