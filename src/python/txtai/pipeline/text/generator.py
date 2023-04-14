"""
Generator module
"""

from ..hfpipeline import HFPipeline


class Generator(HFPipeline):
    """
    Generate text with a causal language model.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None):
        super().__init__(self.task(), path, quantize, gpu, model)

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

    def task(self):
        """
        Get the pipeline task name.

        Returns:
            pipeline task name
        """

        return "text-generation"
