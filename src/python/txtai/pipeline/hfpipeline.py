"""
Hugging Face Transformers pipeline wrapper module
"""

import inspect

from transformers import pipeline

from ..models import Models
from ..util import Resolver

from .tensors import Tensors


class HFPipeline(Tensors):
    """
    Light wrapper around Hugging Face Transformers pipeline component for selected tasks. Adds support for model
    quantization and minor interface changes.
    """

    def __init__(self, task, path=None, quantize=False, gpu=False, model=None, **kwargs):
        """
        Loads a new pipeline model.

        Args:
            task: pipeline task or category
            path: optional path to model, accepts Hugging Face model hub id, local path or (model, tokenizer) tuple.
                  uses default model for task if not provided
            quantize: if model should be quantized, defaults to False
            gpu: True/False if GPU should be enabled, also supports a GPU device id
            model: optional existing pipeline model to wrap
            kwargs: additional keyword arguments to pass to pipeline model
        """

        if model:
            # Check if input model is a Pipeline or a HF pipeline
            self.pipeline = model.pipeline if isinstance(model, HFPipeline) else model
        else:
            # Get device
            deviceid = Models.deviceid(gpu) if "device_map" not in kwargs else None
            device = Models.device(deviceid) if deviceid is not None else None

            # Split into model args, pipeline args
            modelargs, kwargs = self.parseargs(**kwargs)

            # Transformer pipeline task
            if isinstance(path, (list, tuple)):
                # Derive configuration, if possible
                config = path[1] if path[1] and isinstance(path[1], str) else None

                # Load model
                model = Models.load(path[0], config, task)

                self.pipeline = pipeline(task, model=model, tokenizer=path[1], device=device, model_kwargs=modelargs, **kwargs)
            else:
                self.pipeline = pipeline(task, model=path, device=device, model_kwargs=modelargs, **kwargs)

            # Model quantization. Compresses model to int8 precision, improves runtime performance. Only supported on CPU.
            if deviceid == -1 and quantize:
                # pylint: disable=E1101
                self.pipeline.model = self.quantize(self.pipeline.model)

        # Detect unbounded tokenizer typically found in older models
        Models.checklength(self.pipeline.model, self.pipeline.tokenizer)

    def parseargs(self, **kwargs):
        """
        Inspects the pipeline method and splits kwargs into model args and pipeline args.

        Args:
            kwargs: all keyword arguments

        Returns:
            (model args, pipeline args)
        """

        # Get pipeline method arguments
        args = inspect.getfullargspec(pipeline).args

        # Resolve torch dtype, if necessary
        dtype = kwargs.get("torch_dtype")
        if dtype and isinstance(dtype, str) and dtype != "auto":
            kwargs["torch_dtype"] = Resolver()(dtype)

        # Split into modelargs and kwargs
        return ({arg: value for arg, value in kwargs.items() if arg not in args}, {arg: value for arg, value in kwargs.items() if arg in args})

    def maxlength(self):
        """
        Gets the max length to use for generate calls.

        Returns:
            max length
        """

        return Models.maxlength(self.pipeline.model, self.pipeline.tokenizer)
