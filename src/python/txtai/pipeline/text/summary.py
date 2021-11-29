"""
Summary module
"""

import re

from ..hfpipeline import HFPipeline


class Summary(HFPipeline):
    """
    Summarizes text.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None):
        super().__init__("summarization", path, quantize, gpu, model)

    def __call__(self, text, minlength=None, maxlength=None, workers=0):
        """
        Runs a summarization model against a block of text.

        This method supports text as a string or a list. If the input is a string, the return
        type is text. If text is a list, a list of text is returned with a row per block of text.

        Args:
            text: text|list
            minlength: minimum length for summary
            maxlength: maximum length for summary
            workers: number of concurrent workers to use for processing data, defaults to None

        Returns:
            summary text
        """

        # Validate text length greater than max length
        check = maxlength if maxlength else self.pipeline.model.config.max_length

        # Skip text shorter than max length
        texts = text if isinstance(text, list) else [text]
        params = [(x, text if len(text) >= check else None) for x, text in enumerate(texts)]

        kwargs = {"truncation": True}
        if minlength:
            kwargs["min_length"] = minlength
        if maxlength:
            kwargs["max_length"] = maxlength

        inputs = [text for _, text in params if text]
        if inputs:
            # Run summarization pipeline
            results = self.pipeline(inputs, num_workers=workers, **kwargs)

            # Pull out summary text
            results = iter([self.clean(x["summary_text"]) for x in results])
            results = [next(results) if text else texts[x] for x, text in params]
        else:
            # Return original
            results = texts

        return results[0] if isinstance(text, str) else results

    def clean(self, text):
        """
        Applies a series of rules to clean extracted text.

        Args:
            text: input text

        Returns:
            clean text
        """

        text = re.sub(r"\s*\.\s*", ". ", text)
        text = text.strip()

        return text
