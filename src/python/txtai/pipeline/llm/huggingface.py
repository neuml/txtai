"""
Hugging Face module
"""

from threading import Thread

from transformers import AutoModelForImageTextToText, TextIteratorStreamer

from ...models import Models

from ..hfpipeline import HFPipeline

from .generation import Generation


class HFGeneration(Generation):
    """
    Hugging Face Transformers generative model.
    """

    def __init__(self, path, template=None, **kwargs):
        # Call parent constructor
        super().__init__(path, template, **kwargs)

        # Create HuggingFace LLM pipeline
        self.llm = HFLLM(path, **kwargs)

    def isvision(self):
        return isinstance(self.llm.pipeline.model, AutoModelForImageTextToText)

    def stream(self, texts, maxlength, stream, stop, **kwargs):
        yield from self.llm(texts, maxlength=maxlength, stream=stream, stop=stop, **kwargs)


class HFLLM(HFPipeline):
    """
    Hugging Face Transformers large language model (LLM) pipeline. This pipeline autodetects if the model path
    is a text generation or sequence to sequence model.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, task=None, **kwargs):
        super().__init__(self.task(path, task, **kwargs), path, quantize, gpu, model, **kwargs)

        # Load tokenizer, if necessary
        self.pipeline.tokenizer = self.pipeline.tokenizer if self.pipeline.tokenizer else Models.tokenizer(path, **kwargs)

    def __call__(self, text, prefix=None, maxlength=512, workers=0, stream=False, stop=None, **kwargs):
        """
        Generates text. Supports the following input formats:

          - String or list of strings (instruction-tuned models must follow chat templates)
          - List of dictionaries with `role` and `content` key-values or lists of lists

        Args:
            text: text|list
            prefix: optional prefix to prepend to text elements
            maxlength: maximum sequence length
            workers: number of concurrent workers to use for processing data, defaults to None
            stream: stream response if True, defaults to False
            stop: list of stop strings
            kwargs: additional generation keyword arguments

        Returns:
            generated text
        """

        # List of texts
        texts = text if isinstance(text, list) else [text]

        # Add prefix, if necessary
        if prefix:
            texts = [f"{prefix}{x}" for x in texts]

        # Combine all keyword arguments
        args, kwargs = self.parameters(texts, maxlength, workers, stop, **kwargs)

        # Stream response
        if stream:
            return StreamingResponse(self.pipeline, texts, stop, **kwargs)()

        # Run pipeline and extract generated text
        results = [self.extract(result) for result in self.pipeline(*args, **kwargs)]

        return results[0] if isinstance(text, str) else results

    def parameters(self, texts, maxlength, workers, stop, **kwargs):
        """
        Builds a list of arguments and a combined parameter dictionary to use as keyword arguments.

        Args:
            texts: input texts
            maxlength: maximum sequence length
            workers: number of concurrent workers to use for processing data, defaults to None
            stop: list of stop strings
            kwargs: additional generation keyword arguments

        Returns:
            args, kwargs
        """

        # Set defaults and get underlying model
        defaults, model = {"max_length": maxlength, "max_new_tokens": None, "num_workers": workers}, self.pipeline.model

        # Set parameters for vision models and return
        if self.pipeline.task == "image-text-to-text":
            # Maxlength has to be large enough to accomodate images
            defaults["max_length"] = max(maxlength, 2048)

            # Set default token id
            tokenid = model.generation_config.pad_token_id
            model.generation_config.pad_token_id = tokenid if tokenid else model.generation_config.eos_token_id

            # Vision models take all arguments as keyword arguments
            return [], {**{"text": texts, "truncation": True}, **defaults, **kwargs}

        # Add pad token if it's missing from model config
        if not model.config.pad_token_id:
            tokenid = model.config.eos_token_id
            tokenid = tokenid[0] if isinstance(tokenid, list) else tokenid

            # Set pad_token_id parameter
            defaults["pad_token_id"] = tokenid

            # Update tokenizer for batching
            if "batch_size" in kwargs and self.pipeline.tokenizer.pad_token_id is None:
                self.pipeline.tokenizer.pad_token_id = tokenid
                self.pipeline.tokenizer.padding_side = "left"

        # Set tokenizer when stop strings is set
        if stop:
            defaults["tokenizer"] = self.pipeline.tokenizer

        return [texts], {**defaults, **kwargs}

    def extract(self, result):
        """
        Extracts generated text from a pipeline result.

        Args:
            result: pipeline result

        Returns:
            generated text
        """

        # Extract output from list, if necessary
        result = result[0] if isinstance(result, list) else result
        text = result["generated_text"]
        return text[-1]["content"] if isinstance(text, list) else text

    def task(self, path, task, **kwargs):
        """
        Get the pipeline task name.

        Args:
            path: model path input
            task: task name
            kwargs: optional additional keyword arguments

        Returns:
            pipeline task name
        """

        # Mapping from txtai to Hugging Face pipeline tasks
        mapping = {"language-generation": "text-generation", "sequence-sequence": "text2text-generation", "vision": "image-text-to-text"}

        # Attempt to resolve task
        if path and not task:
            task = Models.task(path, **kwargs)

        # Map to Hugging Face task. Default to text2text-generation pipeline when task not resolved.
        return mapping.get(task, "text2text-generation")


class Generator(HFLLM):
    """
    Generate text with a causal language model.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, **kwargs):
        super().__init__(path, quantize, gpu, model, "language-generation", **kwargs)


class Sequences(HFLLM):
    """
    Generate text with a sequence-sequence model.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, **kwargs):
        super().__init__(path, quantize, gpu, model, "sequence-sequence", **kwargs)


class StreamingResponse:
    """
    Generate text as a streaming response.
    """

    def __init__(self, pipeline, texts, stop, **kwargs):
        # Create streamer
        self.stream = TextIteratorStreamer(pipeline.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=5)
        kwargs["streamer"] = self.stream
        kwargs["stop_strings"] = stop

        # Create thread
        self.thread = Thread(target=pipeline, args=[texts], kwargs=kwargs)

        # Store number of inputs
        self.length = len(texts)

    def __call__(self):
        # Start the process
        self.thread.start()

        return self

    def __iter__(self):
        for _ in range(self.length):
            yield from self.stream
