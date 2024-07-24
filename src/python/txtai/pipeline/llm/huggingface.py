"""
Hugging Face module
"""

from threading import Thread

from transformers import TextIteratorStreamer

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

    def stream(self, texts, maxlength, stream, **kwargs):
        yield from self.llm(texts, maxlength=maxlength, stream=stream, **kwargs)


class HFLLM(HFPipeline):
    """
    Hugging Face Transformers large language model (LLM) pipeline. This pipeline autodetects if the model path
    is a text generation or sequence to sequence model.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, task=None, **kwargs):
        super().__init__(self.task(path, task, **kwargs), path, quantize, gpu, model, **kwargs)

        # Load tokenizer, if necessary
        self.pipeline.tokenizer = self.pipeline.tokenizer if self.pipeline.tokenizer else Models.tokenizer(path, **kwargs)

    def __call__(self, text, prefix=None, maxlength=512, workers=0, stream=False, **kwargs):
        """
        Generates text. Supports the following input formats:

          - String or list of strings
          - List of dictionaries with `role` and `content` key-values or lists of lists

        Args:
            text: text|list
            prefix: optional prefix to prepend to text elements
            maxlength: maximum sequence length
            workers: number of concurrent workers to use for processing data, defaults to None
            stream: stream response if True, defaults to False
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
        kwargs = self.parameters(maxlength, workers, **kwargs)

        # Stream response
        if stream:
            return StreamingResponse(self.pipeline, texts, **kwargs)()

        # Run pipeline
        results = self.pipeline(texts, **kwargs)

        # Extract generated text
        results = [self.extract(result) for result in results]

        return results[0] if isinstance(text, str) else results

    def parameters(self, maxlength, workers, **kwargs):
        """
        Builds a combined parameter dictionary.

        Args:
            maxlength: maximum sequence length
            workers: number of concurrent workers to use for processing data, defaults to None
            kwargs: additional generation keyword arguments

        Returns:
            dict of parameters
        """

        # Default parameters
        defaults = {
            "max_length": maxlength,
            "num_workers": workers,
        }

        # Add pad token if it's missing from model config
        model = self.pipeline.model
        if not model.config.pad_token_id:
            tokenid = model.config.eos_token_id
            tokenid = tokenid[0] if isinstance(tokenid, list) else tokenid

            # Set pad_token_id parameter
            defaults["pad_token_id"] = tokenid

            # Update tokenizer for batching
            if "batch_size" in kwargs and self.pipeline.tokenizer.pad_token_id is None:
                self.pipeline.tokenizer.pad_token_id = tokenid
                self.pipeline.tokenizer.padding_side = "left"

        return {**defaults, **kwargs}

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
        mapping = {"language-generation": "text-generation", "sequence-sequence": "text2text-generation"}

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

    def __init__(self, pipeline, texts, **kwargs):
        # Create streamer
        self.stream = TextIteratorStreamer(pipeline.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=5)
        kwargs["streamer"] = self.stream

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
