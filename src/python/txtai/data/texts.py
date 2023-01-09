"""
Texts module
"""

from itertools import chain

from .base import Data


class Texts(Data):
    """
    Tokenizes text datasets as input for training language models.
    """

    def __init__(self, tokenizer, columns, maxlength):
        """
        Creates a new instance for tokenizing Texts training data.

        Args:
            tokenizer: model tokenizer
            columns: tuple of columns to use for text
            maxlength: maximum sequence length
        """

        super().__init__(tokenizer, columns, maxlength)

        # Standardize columns
        if not self.columns:
            self.columns = ("text", None)

    def process(self, data):
        # Column keys
        text1, text2 = self.columns

        # Tokenizer inputs can be single string or string pair, depending on task
        text = (data[text1], data[text2]) if text2 else (data[text1],)

        # Tokenize text and add label
        inputs = self.tokenizer(*text, return_special_tokens_mask=True)

        # Concat and return tokenized inputs
        return self.concat(inputs)

    def concat(self, inputs):
        """
        Concatenates tokenized text into chunks of maxlength.

        Args:
            inputs: tokenized input

        Returns:
            Chunks of tokenized text each with a size of maxlength
        """

        # Concatenate tokenized text
        concat = {k: list(chain(*inputs[k])) for k in inputs.keys()}

        # Calculate total length
        length = len(concat[list(inputs.keys())[0]])

        # Ensure total is multiple of maxlength, drop last incomplete chunk
        if length >= self.maxlength:
            length = (length // self.maxlength) * self.maxlength

        # Split into chunks of maxlength
        result = {k: [v[x : x + self.maxlength] for x in range(0, length, self.maxlength)] for k, v in concat.items()}

        return result
