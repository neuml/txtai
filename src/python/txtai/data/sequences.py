"""
Sequences module
"""

from .base import Data


class Sequences(Data):
    """
    Tokenizes sequence-sequence datasets as input for training sequence-sequence models
    """

    def __init__(self, tokenizer, columns, maxlength, prefix):
        """
        Creates a new instance for tokenizing Sequences training data.

        Args:
            tokenizer: model tokenizer
            columns: tuple of columns to use for text/label
            maxlength: maximum sequence length
            prefix: source prefix
        """

        super().__init__(tokenizer, columns, maxlength)

        # Standardize columns
        if not self.columns:
            self.columns = ("source", "target")

        # Save source prefix
        self.prefix = prefix

    def process(self, data):
        # Column keys
        source, target = self.columns

        # Tokenize source
        source = [self.prefix + x if self.prefix else x for x in data[source]]
        inputs = self.tokenizer(source, max_length=self.maxlength, padding=False, truncation=True)

        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(data[target], max_length=self.maxlength, padding=False, truncation=True)

        # Combine inputs
        inputs["labels"] = targets["input_ids"]

        return inputs
