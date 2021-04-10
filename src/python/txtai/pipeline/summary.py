"""
Summary module
"""

import re

from .hfpipeline import HFPipeline


class Summary(HFPipeline):
    """
    Summarizes text.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None):
        super().__init__("summarization", path, quantize, gpu, model)

    def __call__(self, text, minlength=None, maxlength=None):
        """
        Runs a summarization model against a block of text.

        This method supports text as a string or a list. If the input is a string, the return
        type is text. If text is a list, a list of text is returned with a row per block of text.

        Args:
            text: text|list
            minlength: minimum length for summary
            maxlength: maximum length for summary

        Returns:
            summary text
        """

        kwargs = {"truncation": True}
        if minlength:
            kwargs["min_length"] = minlength
        if maxlength:
            kwargs["max_length"] = maxlength

        # Run summarization pipeline
        results = self.pipeline(text, **kwargs)

        # Convert results to a list if necessary
        if not isinstance(results, list):
            results = [results]

        # Pull out summary text
        results = [self.clean(x["summary_text"]) for x in results]

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
