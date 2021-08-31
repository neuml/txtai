"""
ONNX module
"""

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

        super().__init__(PretrainedConfig())

        # Create ONNX session
        self.model = InferenceSession(model, SessionOptions())

        # Add references for this class to supported AutoModel classes
        name = self.__class__.__name__
        if name not in MODEL_MAPPING:
            MODEL_MAPPING[name] = self.__class__
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING[name] = self.__class__
            MODEL_FOR_QUESTION_ANSWERING_MAPPING[name] = self.__class__

        # Add references for this class to support pipeline AutoTokenizers
        if type(self.config) not in TOKENIZER_MAPPING:
            TOKENIZER_MAPPING[type(self.config)] = None

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
