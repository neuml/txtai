"""
Summary module
"""

import re

from ...models import Models
from ..hfmodel import HFModel


class Summary(HFModel):
    """
    Summarizes text.
    """

    def __init__(self, path=None, quantize=False, gpu=True, batch=64, **kwargs):
        # Default model
        path = path if path else "sshleifer/distilbart-cnn-12-6"

        # Call parent constructor
        super().__init__(path, quantize, gpu, batch)

        # Load model and tokenizer
        self.model, self.tokenizer = self.load(path, "summarization", **kwargs)

    def __call__(self, text, minlength=None, maxlength=None, **kwargs):
        """
        Runs a summarization model against a block of text.

        This method supports text as a string or a list. If the input is a string, the return
        type is text. If text is a list, a list of text is returned with a row per block of text.

        Args:
            text: text|list
            minlength: minimum length for summary
            maxlength: maximum length for summary
            kwargs: additional keyword arguments

        Returns:
            summary text
        """

        # Validate text length greater than max length
        check = maxlength if maxlength else self.maxlength()

        # Skip text shorter than max length
        texts = text if isinstance(text, list) else [text]
        params = [(x, text if len(text) >= check else None) for x, text in enumerate(texts)]

        # Build keyword arguments
        kwargs = self.args(minlength, maxlength)

        inputs = [text for _, text in params if text]
        if inputs:
            # Run summarization
            results = self.generate(inputs, **kwargs)

            # Pull out summary text
            results = iter([self.clean(x) for x in results])
            results = [next(results) if text else texts[x] for x, text in params]
        else:
            # Return original
            results = texts

        return results[0] if isinstance(text, str) else results

    def maxlength(self):
        """
        Gets the max length to use for generate calls.

        Returns:
            max length
        """

        return Models.maxlength(self.model, self.tokenizer)

    def generate(self, inputs, **kwargs):
        """
        Generates outputs.

        Args:
            input: tokenized input
            kwargs: additional keyword arguments

        Returns:
            generated output
        """

        # Prepend prefix, if necessary
        prefix = kwargs.pop("prefix", None)
        if prefix:
            inputs = [prefix + x for x in inputs]

        truncation = kwargs.pop("truncation", False)

        results = []
        for batch in self.batch(inputs, self.batchsize):
            # Tokenize inputs, padding texts of different lengths within a batch
            tokens = self.tokenizer(batch, padding=True, truncation=truncation, return_tensors="pt").to(self.device)

            # Generate outputs
            outputs = self.model.generate(**tokens, **kwargs)

            # Decode outputs
            results.extend(self.tokenizer.decode(x, skip_special_tokens=True) for x in outputs)

        return results

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

    def args(self, minlength, maxlength):
        """
        Builds keyword arguments.

        Args:
            minlength: minimum length for summary
            maxlength: maximum length for summary

        Returns:
            keyword arguments
        """

        # Defaults
        kwargs = {"truncation": True, "num_beams": 4, "max_new_tokens": 256}

        # Get default summary task parameters
        params = getattr(self.model.config, "task_specific_params", None)
        if params:
            params = params.get("summarization", {})

            # Ignore max length
            kwargs.update({k: v for k, v in params.items() if k not in ["max_length"]})

        if minlength:
            kwargs["min_length"] = minlength
        if maxlength:
            kwargs["max_length"] = maxlength
            kwargs["max_new_tokens"] = None

            # Default minlength if not provided or it's bigger than maxlength
            if "min_length" not in kwargs or kwargs["min_length"] > kwargs["max_length"]:
                kwargs["min_length"] = kwargs["max_length"]

        return kwargs
