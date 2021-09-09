"""
Labels module
"""

from .base import Data


class Labels(Data):
    """
    Tokenizes text-classification datasets used as input for training text-classification models.
    """

    def __init__(self, tokenizer, columns, maxlength):
        """
        Creates a new instance for tokenizing Labels training data.

        Args:
            tokenizer: model tokenizer
            columns: tuple of columns to use for text/label
            maxlength: maximum sequence length
        """

        super().__init__(tokenizer, columns, maxlength)

        # Standardize columns
        if not self.columns:
            self.columns = ("text", None, "label")
        elif len(columns) < 3:
            self.columns = (self.columns[0], None, self.columns[-1])

    def process(self, data):
        # Column keys
        text1, text2, label = self.columns

        # Tokenizer inputs can be single string or string pair, depending on task
        text = (data[text1], data[text2]) if text2 else (data[text1],)

        # Tokenize text and add label
        inputs = self.tokenizer(*text, max_length=self.maxlength, padding=True, truncation=True)
        inputs[label] = data[label]

        return inputs
