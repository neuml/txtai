"""
Machine learning model to ONNX export module
"""

from ..base import Pipeline

try:
    from onnxmltools import convert_sklearn
    from onnxmltools.convert.common.data_types import StringTensorType

    from skl2onnx.helpers.onnx_helper import save_onnx_model, select_model_inputs_outputs

    ONNX_MLTOOLS = True
except ImportError:
    ONNX_MLTOOLS = False


class MLOnnx(Pipeline):
    """
    Exports a machine learning model to ONNX using ONNXMLTools.
    """

    def __init__(self):
        """
        Creates a new MLOnnx pipeline.
        """

        if not ONNX_MLTOOLS:
            raise ImportError('MLOnnx pipeline is not available - install "pipeline" extra to enable')

    def __call__(self, model, task="default", output=None, opset=12):
        """
        Exports a machine learning model to ONNX using ONNXMLTools.

        Args:
            model: model to export
            task: optional model task or category
            output: optional output model path, defaults to return byte array if None
            opset: onnx opset, defaults to 12

        Returns:
            path to model output or model as bytes depending on output parameter
        """

        # Convert scikit-learn model to ONNX
        model = convert_sklearn(model, task, initial_types=[("input_ids", StringTensorType([None, None]))], target_opset=opset)

        # Prune model graph down to only output probabilities
        model = select_model_inputs_outputs(model, outputs="probabilities")

        # pylint: disable=E1101
        # Rename output to logits for consistency with other models
        model.graph.output[0].name = "logits"
        model.graph.node[0].output[0] = "logits"

        # Save model to specified output path or return bytes
        model = save_onnx_model(model, output)
        return output if output else model
