"""
Texts module
"""

from itertools import chain

from .base import Data


class Texts(Data):
    """
    Tokenizes text datasets as input for training language models.
    """

    def __init__(self, tokenizer, columns, maxlength, merge):
        """
        Creates a new instance for tokenizing Texts training data.

        Args:
            tokenizer: model tokenizer
            columns: tuple of columns to use for text
            maxlength: maximum sequence length
            merge: determines how chunks are combined for language modeling tasks - "concat" (default), "pack" or None
        """

        super().__init__(tokenizer, columns, maxlength)

        # Standardize columns
        if not self.columns:
            self.columns = ("text", None)

        # Method to combine chunks
        self.merge = merge

    def process(self, data):
        # Column keys
        text1, text2 = self.columns

        # Tokenizer inputs can be single string or string pair, depending on task
        text = (data[text1], data[text2]) if text2 else (data[text1],)

        # Tokenize text and add label
        inputs = self.tokenizer(*text, return_special_tokens_mask=True)

        # Combine inputs based on parameters
        return self.concat(inputs) if self.merge == "concat" else self.pack(inputs) if self.merge == "pack" else inputs

    def concat(self, inputs):
        """
        Concatenates tokenized text into chunks of maxlength. This method guarantees that each chunk is maxlength
        size and splits data across multiple chunks if needed.

        This is best with general language modeling tasks like masked language modeling that are streams of text.

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

    def pack(self, inputs):
        """
        Packs tokenized text into chunks up to maxlength. This method guarantees that data is never split across
        multiple chunks.

        This is best with instruction/prompt learning where it's crucial to ensure entire records are preserved.

        Args:
            inputs: tokenized input

        Returns:
            Chunks of tokenized text each with a size of maxlength
        """

        # Sort lists by length descending
        inputs = {k: sorted(v, key=len, reverse=True) for k, v in inputs.items()}

        # Inputs has lists of equal length per column
        columns = list(inputs.keys())

        # Create empty results dict
        results = {column: [] for column in columns}

        # Iterate over values in first column since all column lengths per row are equal
        length, index, rows = 0, 0, inputs[columns[0]]
        for x, row in enumerate(rows):
            length += len(row)
            nextlength = len(rows[x + 1]) if x < len(rows) - 1 else 0

            # New row
            if (length + nextlength) >= self.maxlength:
                for column in columns:
                    results[column].append(list(chain(*inputs[column][index : x + 1])))

                # Reset length and index
                length, index = 0, x + 1

        # Last row
        if length:
            for column in columns:
                results[column].append(list(chain(*inputs[column][index : len(rows)])))

        return results
