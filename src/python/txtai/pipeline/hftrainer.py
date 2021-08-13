"""
Hugging Face Transformers trainer wrapper module
"""

import sys

import torch

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, set_seed

from transformers import TrainingArguments as HFTrainingArguments

from .tensors import Tensors
from ..models import Models


class HFTrainer(Tensors):
    """
    Trains a new Hugging Face Transformer model using the Trainer framework.
    """

    def __call__(self, base, train, validation=None, columns=None, maxlength=None, **args):
        """
        Builds a new model using arguments.

        Args:
            base: base model - supports a file path or (model, tokenizer) tuple
            train: training data
            validation: validation data
            columns: tuple of columns to use for text/label, defaults to (text, None, label)
            maxlength: maximum sequence length, defaults to tokenizer.model_max_length
            args: training arguments

        Returns:
            (model, tokenizer)
        """

        # Parse TrainingArguments
        args = self.parse(args)

        # Set seed for model reproducibility
        set_seed(args.seed)

        # Load model configuration, tokenizer and max sequence length
        config, tokenizer, maxlength = self.load(base, maxlength)

        # Prepare datasets
        train, validation, labels = self.datasets(train, validation, columns, tokenizer, maxlength)

        # Create model to train
        model = self.model(base, config, labels)

        # Build trainer
        trainer = Trainer(model=model, tokenizer=tokenizer, args=args, train_dataset=train, eval_dataset=validation if validation else None)

        # Run training
        trainer.train()

        # Run evaluation
        if validation:
            trainer.evaluate()

        # Save model outputs
        if args.should_save:
            trainer.save_model()
            trainer.save_state()

        # Put model in eval mode to disable weight updates and return (model, tokenizer)
        return (model.eval(), tokenizer)

    def parse(self, updates):
        """
        Parses and merges custom arguments with defaults.

        Args:
            updates: custom arguments

        Returns:
            TrainingArguments
        """

        # Default training arguments
        args = {"output_dir": "", "save_strategy": "no", "report_to": "none", "log_level": "warning"}

        # Apply custom arguments
        args.update(updates)

        return TrainingArguments(**args)

    def load(self, base, maxlength):
        """
        Loads the base config and tokenizer.

        Args:
            base: base model - supports a file path or (model, tokenizer) tuple
            maxlength: maximum sequence length

        Returns:
            (config, tokenizer, maxlength)
        """

        if isinstance(base, tuple):
            # Unpack existing config and tokenizer
            model, tokenizer = base
            config = model.config
        else:
            # Load config
            config = AutoConfig.from_pretrained(base)

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base)

        # Detect unbounded tokenizer
        Models.checklength(config, tokenizer)

        # Derive max sequence length
        maxlength = min(maxlength if maxlength else sys.maxsize, tokenizer.model_max_length)

        return (config, tokenizer, maxlength)

    def datasets(self, train, validation, columns, tokenizer, maxlength):
        """
        Prepares training and validation datasets for training.

        Args:
            train: training data
            validation: validation data
            columns: tuple of columns to use for text/label
            tokenizer: model tokenizer
            maxlength: maximum sequence length

        Returns:
            (train, validation, labels)
        """

        # Standardize columns
        if not columns:
            columns = ("text", None, "label")
        elif len(columns) < 3:
            columns = (columns[0], None, columns[-1])

        # Prepare training data and labels
        train, labels = self.prepare(train, columns, tokenizer, maxlength)

        # Prepare validation data
        if validation:
            validation, _ = self.prepare(validation, columns, tokenizer, maxlength)

        return (train, validation, labels)

    def prepare(self, data, columns, tokenizer, maxlength):
        """
        Prepares and tokenizes data for training.

        Args:
            data: input data
            columns: tuple of columns to use for text/label
            tokenizer: model tokenizer
            maxlength: maximum sequence length

        Returns:
            (tokens, labels)
        """

        if hasattr(data, "map"):
            # Hugging Face dataset
            tokens = data.map(lambda row: self.tokenize(row, columns, tokenizer, maxlength))
            labels = sorted(data.unique(columns[-1]))
        else:
            # pandas DataFrame
            if hasattr(data, "to_dict"):
                data = data.to_dict("records")

            # Process list of dicts
            tokens = TokenDataset([self.tokenize(row, columns, tokenizer, maxlength) for row in data])
            labels = sorted({row[columns[-1]] for row in data})

        # Determine number of labels, account for regression tasks
        labels = 1 if [x for x in labels if isinstance(x, float)] else len(labels)

        return (tokens, labels)

    def tokenize(self, row, columns, tokenizer, maxlength):
        """
        Tokenizes data row.

        Args:
            row: input data
            columns: tuple of columns to use for text/label
            tokenizer: model tokenizer
            maxlength: maximum sequence length

        Returns:
            tokens
        """

        # Column keys
        text1, text2, label = columns

        # Tokenizer inputs can be single string or string pair, depending on task
        text = (row[text1], row[text2]) if text2 else (row[text1],)

        # Tokenize text and add label
        inputs = tokenizer(*text, max_length=maxlength, padding=True, truncation=True)
        inputs[label] = row[label]

        return inputs

    def model(self, base, config, labels):
        """
        Loads the base model to train.

        Args:
            base: base model - supports a file path or (model, tokenizer) tuple
            config: model configuration
            labels: number of labels

        Returns:
            model
        """

        # Add number of labels to config
        config.update({"num_labels": labels})

        # Unpack existing model or create new model from config
        return base[0] if isinstance(base, tuple) else AutoModelForSequenceClassification.from_pretrained(base, config=config)


class TokenDataset(torch.utils.data.Dataset):
    """
    Default dataset used to hold tokenized data.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class TrainingArguments(HFTrainingArguments):
    """
    Extends standard TrainingArguments to make the output directory optional for transient models.
    """

    @property
    def should_save(self):
        """
        Override should_save to disable model saving when output directory is None.

        Returns:
            If model should be saved
        """

        return super().should_save if self.output_dir else False
