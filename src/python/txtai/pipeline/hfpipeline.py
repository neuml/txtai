"""
Hugging Face Transformers pipeline wrapper module
"""

from transformers import pipeline

from ..models import Models
from .tensors import Tensors


class HFPipeline(Tensors):
    """
    Light wrapper around Hugging Face Transformers pipeline component for selected tasks. Adds support for model
    quantization and minor interface changes.
    """

    def __init__(self, task, path=None, quantize=False, gpu=False, model=None):
        """
        Loads a new pipeline model.

        Args:
            task: pipeline task or category
            path: optional path to model, accepts Hugging Face model hub id, local path or (model, tokenizer) tuple.
                  uses default model for task if not provided
            quantize: if model should be quantized, defaults to False
            gpu: True/False if GPU should be enabled, also supports a GPU device id
            model: optional existing pipeline model to wrap
        """

        if model:
            # Check if input model is a Pipeline or a HF pipeline
            self.pipeline = model.pipeline if isinstance(model, HFPipeline) else model
        else:
            # Get device id
            deviceid = Models.deviceid(gpu)

            # Transformer pipeline task
            if isinstance(path, (list, tuple)):
                # Derive configuration, if possible
                config = path[1] if path[1] and isinstance(path[1], str) else None

                self.pipeline = pipeline(task, model=Models.load(path[0], config, task), tokenizer=path[1], device=deviceid)
            else:
                self.pipeline = pipeline(task, model=path, tokenizer=path, device=deviceid)

            # Model quantization. Compresses model to int8 precision, improves runtime performance. Only supported on CPU.
            if deviceid == -1 and quantize:
                # pylint: disable=E1101
                self.pipeline.model = self.quantize(self.pipeline.model)

        # Detect unbounded tokenizer typically found in older models
        Models.checklength(self.pipeline.model, self.pipeline.tokenizer)
