"""
Hugging Face Transformers ONNX export module
"""

from collections import OrderedDict
from io import BytesIO
from itertools import chain
from tempfile import NamedTemporaryFile

# Conditional import
try:
    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
    from onnxruntime.quantization import quantize_dynamic

    ONNX_RUNTIME = True
except ImportError:
    ONNX_RUNTIME = False

from torch.onnx import export

from transformers import AutoModel, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer

from ...models import MeanPooling
from ..tensors import Tensors


class HFOnnx(Tensors):
    """
    Exports a Hugging Face Transformer model to ONNX.
    """

    def __call__(self, path, task="default", output=None, quantize=False, opset=12):
        """
        Exports a Hugging Face Transformer model to ONNX.

        Args:
            path: path to model, accepts Hugging Face model hub id, local path or (model, tokenizer) tuple
            task: optional model task or category, determines the model type and outputs, defaults to export hidden state
            output: optional output model path, defaults to return byte array if None
            quantize: if model should be quantized (requires onnx to be installed), defaults to False
            opset: onnx opset, defaults to 12

        Returns:
            path to model output or model as bytes depending on output parameter
        """

        inputs, outputs, model = self.parameters(task)

        if isinstance(path, (list, tuple)):
            model, tokenizer = path
            model = model.cpu()
        else:
            model = model(path)
            tokenizer = AutoTokenizer.from_pretrained(path)

        # Generate dummy inputs
        dummy = dict(tokenizer(["test inputs"], return_tensors="pt"))

        # Default to BytesIO if no output file provided
        output = output if output else BytesIO()

        # Export model to ONNX
        export(
            model,
            (dummy,),
            output,
            opset_version=opset,
            do_constant_folding=True,
            input_names=list(inputs.keys()),
            output_names=list(outputs.keys()),
            dynamic_axes=dict(chain(inputs.items(), outputs.items())),
        )

        # Quantize model
        if quantize:
            if not ONNX_RUNTIME:
                raise ImportError('onnxruntime is not available - install "pipeline" extra to enable')

            output = self.quantization(output)

        if isinstance(output, BytesIO):
            # Reset stream and return bytes
            output.seek(0)
            output = output.read()

        return output

    def quantization(self, output):
        """
        Quantizes an ONNX model.

        Args:
            output: path to ONNX model or BytesIO with model data

        Returns:
            quantized model as file path or bytes
        """

        temp = None
        if isinstance(output, BytesIO):
            with NamedTemporaryFile(suffix=".quant", delete=False) as tmpfile:
                temp = tmpfile.name

            with open(temp, "wb") as f:
                f.write(output.getbuffer())

            output = temp

        # Optimize model - only need CPU provider
        sess_option = SessionOptions()
        sess_option.optimized_model_filepath = output
        sess_option.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
        _ = InferenceSession(output, sess_option, ["CPUExecutionProvider"])

        # Quantize optimized model
        quantize_dynamic(output, output, optimize_model=False)

        # Read file back to bytes if temp file was created
        if temp:
            with open(temp, "rb") as f:
                output = f.read()

        return output

    def parameters(self, task):
        """
        Defines inputs and outputs for an ONNX model.

        Args:
            task: task name used to lookup model configuration

        Returns:
            (inputs, outputs, model function)
        """

        inputs = OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
                ("token_type_ids", {0: "batch", 1: "sequence"}),
            ]
        )

        config = {
            "default": (OrderedDict({"last_hidden_state": {0: "batch", 1: "sequence"}}), AutoModel.from_pretrained),
            "pooling": (OrderedDict({"embeddings": {0: "batch", 1: "sequence"}}), lambda x: MeanPoolingOnnx(x, -1)),
            "question-answering": (
                OrderedDict(
                    {
                        "start_logits": {0: "batch", 1: "sequence"},
                        "end_logits": {0: "batch", 1: "sequence"},
                    }
                ),
                AutoModelForQuestionAnswering.from_pretrained,
            ),
            "text-classification": (OrderedDict({"logits": {0: "batch"}}), AutoModelForSequenceClassification.from_pretrained),
        }

        # Aliases
        config["zero-shot-classification"] = config["text-classification"]

        return (inputs,) + config[task]


class MeanPoolingOnnx(MeanPooling):
    """
    Extends MeanPooling class to name inputs to model, which is required
    to export to ONNX.
    """

    # pylint: disable=W0221
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # Build list of arguments dynamically since some models take token_type_ids
        # and others don't
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        return super().forward(**inputs)
