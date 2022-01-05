"""
ONNX module
"""

# Conditional import
try:
    import onnxruntime as ort

    ONNX_RUNTIME = True
except ImportError:
    ONNX_RUNTIME = False

import numpy as np
import torch

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel

from .registry import Registry

# pylint: disable=W0223
class OnnxModel(PreTrainedModel):
    """
    Provides a Transformers/PyTorch compatible interface for ONNX models. Handles casting inputs
    and outputs with minimal to no copying of data.
    """

    def __init__(self, model, config=None):
        """
        Creates a new OnnxModel.

        Args:
            model: path to model or InferenceSession
            config: path to model configuration
        """

        if not ONNX_RUNTIME:
            raise ImportError('onnxruntime is not available - install "model" extra to enable')

        super().__init__(AutoConfig.from_pretrained(config) if config else OnnxConfig())

        # Create ONNX session
        self.model = ort.InferenceSession(model, ort.SessionOptions(), self.providers())

        # Add references for this class to supported AutoModel classes
        Registry.register(self)

    def providers(self):
        """
        Returns a list of available and usable providers.

        Returns:
            list of available and usable providers
        """

        # Create list of providers, prefer CUDA provider if available
        # CUDA provider only available if GPU is available and onnxruntime-gpu installed
        if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Default when CUDA provider isn't available
        return ["CPUExecutionProvider"]

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
        # Detect if logits is an output and return classifier output in that case
        if any(x.name for x in self.model.get_outputs() if x.name == "logits"):
            return SequenceClassifierOutput(logits=torch.from_numpy(np.array(results[0])))

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
