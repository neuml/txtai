"""
Data module
"""

from .tokens import Tokens


class Data:
    """
    Base data tokenization class.
    """

    def __init__(self, tokenizer, columns, maxlength):
        """
        Creates new base instance for tokenizing data.

        Args:
            tokenizer: model tokenizer
            columns: column names
            maxlength: maximum sequence length
        """

        self.tokenizer = tokenizer
        self.columns = columns
        self.maxlength = maxlength

    def __call__(self, train, validation):
        """
        Tokenizes training and validation data and returns processed datasets.

        Args:
            train: training data
            validation: validation data

        Returns:
            (train, validation)
        """

        return (self.prepare(train, self.process), self.prepare(validation, self.process) if validation else None)

    def prepare(self, data, fn):
        """
        Prepares and tokenizes data for training.

        Args:
            data: input data
            fn: tokenize processing function to apply

        Returns:
            tokens
        """

        if hasattr(data, "map"):
            # Hugging Face dataset
            tokens = data.map(fn, batched=True, remove_columns=data.column_names)
        else:
            # pandas DataFrame
            if hasattr(data, "to_dict"):
                data = data.to_dict("records")

            # Re-orient data into columns for efficient batch tokenization
            columns = {}
            for row in data:
                for column in row.keys():
                    if column not in columns:
                        columns[column] = []

                    columns[column].append(row[column])

            # Process column-oriented data
            tokens = Tokens(fn(columns))

        return tokens

    def labels(self, data):
        """
        Extracts a list of unique labels from data.

        Args:
            data: input data

        Returns:
            list of unique labels
        """

        if hasattr(data, "map"):
            # Hugging Face dataset
            labels = sorted(data.unique(self.columns[-1]))
        else:
            # pandas DataFrame
            if hasattr(data, "to_dict"):
                data = data.to_dict("records")

            # Process list of dicts
            labels = sorted({row[self.columns[-1]] for row in data})

        # Determine number of labels, account for regression tasks
        return 1 if [x for x in labels if isinstance(x, float)] else len(labels)

    def process(self, data):
        """
        Tokenizes batch of input data

        Args:
            data: input data batch

        Returns:
            tokenized data
        """

        return data
