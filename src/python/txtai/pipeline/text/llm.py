"""
LLM Module
"""

from ...models import Models

from ..hfpipeline import HFPipeline


class LLM(HFPipeline):
    """
    Runs prompts through a large language model (LLM). This pipeline autodetects if the model path is a text generation or
    sequence to sequence model.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, task=None, **kwargs):
        super().__init__(self.task(path, task, **kwargs), path if path else "google/flan-t5-base", quantize, gpu, model, **kwargs)

        # Load tokenizer, if necessary
        self.pipeline.tokenizer = self.pipeline.tokenizer if self.pipeline.tokenizer else Models.tokenizer(path, **kwargs)

    def __call__(self, text, prefix=None, maxlength=512, workers=0, **kwargs):
        """
        Generates text using input text

        Args:
            text: text|list
            prefix: optional prefix to prepend to text elements
            maxlength: maximum sequence length
            workers: number of concurrent workers to use for processing data, defaults to None
            kwargs: additional generation keyword arguments

        Returns:
            generated text
        """

        # List of texts
        texts = text if isinstance(text, list) else [text]

        # Add prefix, if necessary
        if prefix:
            texts = [f"{prefix}{x}" for x in texts]

        # Run pipeline
        results = self.pipeline(texts, max_length=maxlength, num_workers=workers, **kwargs)

        # Get generated text
        results = [self.clean(texts[x], result) for x, result in enumerate(results)]

        return results[0] if isinstance(text, str) else results

    def clean(self, prompt, result):
        """
        Applies a series of rules to clean generated text.

        Args:
            prompt: original input prompt
            result: input result

        Returns:
            clean text
        """

        # Extract output from list, if necessary
        result = result[0] if isinstance(result, list) else result

        # Get generated text field
        text = result["generated_text"]

        # Replace input prompt
        text = text.replace(prompt, "")

        # Apply text cleaning rules
        return text.replace("$=", "<=").strip()

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
