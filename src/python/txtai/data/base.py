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

    def __call__(self, train, validation, workers):
        """
        Tokenizes training and validation data and returns processed datasets.

        Args:
            train: training data
            validation: validation data
            workers: number of concurrent tokenizers when processing datasets, only main process used when set to None

        Returns:
            (train, validation)
        """

        return (self.prepare(train, self.process, workers), self.prepare(validation, self.process, workers) if validation else None)

    def prepare(self, data, fn, workers):
        """
        Prepares and tokenizes data for training.

        Args:
            data: input data
            fn: tokenize processing function to apply
            workers: number of concurrent tokenizers when processing datasets, only main process used when set to None

        Returns:
            tokens
        """

        if hasattr(data, "map"):
            # Hugging Face dataset
            tokens = data.map(fn, batched=True, num_proc=workers, remove_columns=data.column_names)
        else:
            # Re-orient data into columns for efficient batch tokenization
            columns = {}
            if hasattr(data, "columns"):
                # Polars/pandas DataFrame
                for column in data.columns:
                    columns[column] = list(data[column])
            else:
                # Iterable dicts
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

        # Last column is label
        column = self.columns[-1]

        # Return length of labels if it's an array
        length = self.length(data[column][0] if hasattr(data, "columns") else data[0][column])
        if length:
            return length

        if hasattr(data, "map"):
            # Hugging Face dataset
            labels = sorted(data.unique(self.columns[-1]))
        elif hasattr(data, "columns"):
            # Polars/pandas DataFrame
            labels = sorted(data[self.columns[-1]].unique())
        else:
            # Iterable dicts
            labels = sorted({row[self.columns[-1]] for row in data})

        # Labels are single numeric values per entry
        #   - Consider a regression task if at least one label isn't an integer
        #   - Otherwise use number of labels for a classification task
        return 1 if [x for x in labels if float(x) != int(x)] else len(labels)

    def process(self, data):
        """
        Tokenizes batch of input data

        Args:
            data: input data batch

        Returns:
            tokenized data
        """

        return data

    def length(self, value):
        """
        Returns the length of value if value has a len function defined. Otherwise,
        None is returned.

        Args:
            value: value to check

        Returns:
            length of value if available, otherwise returns None
        """

        return len(value) if hasattr(value, "__len__") else None
