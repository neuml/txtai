"""
Hugging Face pipeline wrapper module
"""

import torch

from transformers import pipeline

from .base import Pipeline


class HFPipeline(Pipeline):
    """
    Light wrapper around Hugging Face's pipeline component for selected tasks. Adds support for model
    quantization and minor interface changes.
    """

    def __init__(self, task, path=None, quantize=False, gpu=False, model=None):
        """
        Loads a new pipeline model.

        Args:
            task: pipeline task or category
            path: optional path to model, accepts Hugging Face model hub id or local path,
                  uses default model for task if not provided
            quantize: if model should be quantized, defaults to False
            gpu: if gpu inference should be used (only works if GPUs are available)
            model: optional existing pipeline model to wrap
        """

        if model:
            # Check if input model is a Pipeline or a HF pipeline
            self.pipeline = model.pipeline if isinstance(model, Pipeline) else model
        else:
            # Enable GPU inference if explicitly set and a GPU is available
            gpu = gpu and torch.cuda.is_available()

            # Transformer pipeline task
            self.pipeline = pipeline(task, model=path, tokenizer=path, device=0 if gpu else -1)

            # Model quantization. Compresses model to int8 precision, improves runtime performance. Only supported on CPU.
            if not gpu and quantize:
                # pylint: disable=E1101
                self.pipeline.model = torch.quantization.quantize_dynamic(self.pipeline.model, {torch.nn.Linear}, dtype=torch.qint8)
