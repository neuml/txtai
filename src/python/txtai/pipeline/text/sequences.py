"""
Sequences module
"""

from ..hfpipeline import HFPipeline


class Sequences(HFPipeline):
    """
    Runs text through a sequence-sequence model.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None):
        super().__init__("text2text-generation", path, quantize, gpu, model)

    def __call__(self, text, prefix=None, maxlength=512, workers=0):
        """
        Runs a sequence-sequence model for input texts.

        Args:
            text: text|list
            prefix: optional prefix to prepend to text elements
            maxlength: maximum sequence length
            workers: number of concurrent workers to use for processing data, defaults to None

        Returns:
            generated text
        """

        # List of texts
        texts = text if isinstance(text, list) else [text]

        # Add prefix, if necessary
        if prefix:
            texts = [f"{prefix}{x}" for x in texts]

        # Run text2text pipeline
        results = self.pipeline(texts, max_length=maxlength, num_workers=workers)

        # Get generated text
        results = [self.clean(x["generated_text"]) for x in results]

        return results[0] if isinstance(text, str) else results

    def clean(self, text):
        """
        Applies a series of rules to clean generated text.

        Args:
            text: input text

        Returns:
            clean text
        """

        return text.replace("$=", "<=")
